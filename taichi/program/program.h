// Program  - Taichi program execution context

#pragma once

#include <functional>
#include <optional>
#include <atomic>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/snode.h"
#include "taichi/lang_util.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/backends/metal/metal_program.h"
#include "taichi/backends/cc/cc_program.h"
#include "taichi/program/callable.h"
#include "taichi/program/aot_module_builder.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/snode_rw_accessors_bank.h"
#include "taichi/program/context.h"
#include "taichi/runtime/runtime.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/system/memory_pool.h"
#include "taichi/system/threading.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/program/sparse_matrix.h"

namespace taichi {
namespace lang {

struct JITEvaluatorId {
  std::thread::id thread_id;
  // Note that on certain backends (e.g. CUDA), functions created in one
  // thread cannot be used in another. Hence the thread_id member.
  int op;
  DataType ret, lhs, rhs;
  bool is_binary;

  UnaryOpType unary_op() const {
    TI_ASSERT(!is_binary);
    return (UnaryOpType)op;
  }

  BinaryOpType binary_op() const {
    TI_ASSERT(is_binary);
    return (BinaryOpType)op;
  }

  bool operator==(const JITEvaluatorId &o) const {
    return thread_id == o.thread_id && op == o.op && ret == o.ret &&
           lhs == o.lhs && rhs == o.rhs && is_binary == o.is_binary;
  }
};

}  // namespace lang
}  // namespace taichi

namespace std {
template <>
struct hash<taichi::lang::JITEvaluatorId> {
  std::size_t operator()(
      taichi::lang::JITEvaluatorId const &id) const noexcept {
    return ((std::size_t)id.op | (id.ret.hash() << 8) | (id.lhs.hash() << 16) |
            (id.rhs.hash() << 24) | ((std::size_t)id.is_binary << 31)) ^
           (std::hash<std::thread::id>{}(id.thread_id) << 32);
  }
};
}  // namespace std

namespace taichi {
namespace lang {

extern Program *current_program;

TI_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class StructCompiler;

class AsyncEngine;

/**
 * Note [Backend-specific ProgramImpl]
 * We're working in progress to keep Program class minimal and move all backend
 * specific logic to their corresponding backend ProgramImpls.

 * If you are thinking about exposing/adding attributes/methods to Program
 class,
 * please first think about if it's general for all backends:
 * - If so, please consider adding it to ProgramImpl class first.
 * - Otherwise please add it to a backend-specific ProgramImpl, e.g.
 * LlvmProgramImpl, MetalProgramImpl..
 */

class Program {
 public:
  using Kernel = taichi::lang::Kernel;
  Callable *current_callable{nullptr};
  CompileConfig config;
  bool sync{false};  // device/host synchronized?
  std::unique_ptr<MemoryPool> memory_pool{nullptr};
  uint64 *result_buffer{nullptr};  // Note result_buffer is used by all backends

  std::unordered_map<int, SNode *>
      snodes;  // TODO: seems LLVM specific but used by state_flow_graph.cpp.

  std::unique_ptr<AsyncEngine> async_engine{nullptr};

  std::vector<std::unique_ptr<Kernel>> kernels;

  std::unique_ptr<KernelProfilerBase> profiler{nullptr};

  std::unordered_map<JITEvaluatorId, std::unique_ptr<Kernel>>
      jit_evaluator_cache;
  std::mutex jit_evaluator_cache_mut;

  // Note: for now we let all Programs share a single TypeFactory for smooth
  // migration. In the future each program should have its own copy.
  static TypeFactory &get_type_factory();

  Program() : Program(default_compile_config.arch) {
  }

  explicit Program(Arch arch);

  ~Program();

  void print_kernel_profile_info() {
    profiler->print();
  }

  struct KernelProfilerQueryResult {
    int counter{0};
    double min{0.0};
    double max{0.0};
    double avg{0.0};
  };

  KernelProfilerQueryResult query_kernel_profile_info(const std::string &name) {
    KernelProfilerQueryResult query_result;
    profiler->query(name, query_result.counter, query_result.min,
                    query_result.max, query_result.avg);
    return query_result;
  }

  void clear_kernel_profile_info() {
    profiler->clear();
  }

  void profiler_start(const std::string &name) {
    profiler->start(name);
  }

  void profiler_stop() {
    profiler->stop();
  }

  KernelProfilerBase *get_profiler() {
    return profiler.get();
  }

  void synchronize();

  // See AsyncEngine::flush().
  // Only useful when async mode is enabled.
  void async_flush();

  /**
   * Materializes the runtime.
   */
  void materialize_runtime();

  int get_snode_tree_size();

  void visualize_layout(const std::string &fn);

  Kernel &kernel(const std::function<void()> &body,
                 const std::string &name = "",
                 bool grad = false) {
    // Expr::set_allow_store(true);
    auto func = std::make_unique<Kernel>(*this, body, name, grad);
    // Expr::set_allow_store(false);
    kernels.emplace_back(std::move(func));
    return *kernels.back();
  }

  Function *create_function(const FunctionKey &func_key);

  // TODO: This function is doing two things: 1) compiling CHI IR, and 2)
  // offloading them to each backend. We should probably separate the logic?
  // TODO: Optional offloaded is used by async mode, we might refactor it in the
  // future.
  FunctionType compile(Kernel &kernel, OffloadedStmt *offloaded = nullptr);

  void check_runtime_error();

  Kernel &get_snode_reader(SNode *snode);

  Kernel &get_snode_writer(SNode *snode);

  uint64 fetch_result_uint64(int i);

  template <typename T>
  T fetch_result(int i) {
    return taichi_union_cast_with_different_sizes<T>(fetch_result_uint64(i));
  }

  Arch get_host_arch() {
    return host_arch();
  }

  Arch get_snode_accessor_arch();

  float64 get_total_compilation_time() {
    return total_compilation_time_;
  }

  void finalize();

  static int get_kernel_id() {
    static int id = 0;
    TI_ASSERT(id < 100000);
    return id++;
  }

  static int default_block_dim(const CompileConfig &config);

  // Note this method is specific to LlvmProgramImpl, but we keep it here since
  // it's exposed to python.
  void print_memory_profiler_info();

  // Returns zero if the SNode is statically allocated
  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  inline SNodeGlobalVarExprMap *get_snode_to_glb_var_exprs() {
    return &snode_to_glb_var_exprs_;
  }

  inline SNodeRwAccessorsBank &get_snode_rw_accessors_bank() {
    return snode_rw_accessors_bank_;
  }

  /**
   * Destroys a new SNode tree.
   *
   * @param snode_tree The pointer to SNode tree.
   */
  void destroy_snode_tree(SNodeTree *snode_tree);

  /**
   * Adds a new SNode tree.
   *
   * @param root The root of the new SNode tree.
   * @return The pointer to SNode tree.
   */
  SNodeTree *add_snode_tree(std::unique_ptr<SNode> root);

  /**
   * Gets the root of a SNode tree.
   *
   * @param tree_id Index of the SNode tree
   * @return Root of the tree
   */
  SNode *get_snode_root(int tree_id);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(Arch arch);

  LlvmProgramImpl *get_llvm_program_impl() {
    return static_cast<LlvmProgramImpl *>(program_impl_.get());
  }

 private:
  /**
   * Materializes a new SNodeTree.
   *
   * JIT compiles the @param tree to backend-specific data types.
   */
  void materialize_snode_tree(SNodeTree *tree);

  // SNode information that requires using Program.
  SNodeGlobalVarExprMap snode_to_glb_var_exprs_;
  SNodeRwAccessorsBank snode_rw_accessors_bank_;

  std::vector<std::unique_ptr<SNodeTree>> snode_trees_;

  std::vector<std::unique_ptr<Function>> functions_;
  std::unordered_map<FunctionKey, Function *> function_map_;

  std::unique_ptr<ProgramImpl> program_impl_;
  float64 total_compilation_time_{0.0};
  static std::atomic<int> num_instances_;
  bool finalized_{false};

 public:
#ifdef TI_WITH_CC
  // C backend related data structures
  std::unique_ptr<cccp::CCProgram> cc_program;
#endif
};

}  // namespace lang
}  // namespace taichi
