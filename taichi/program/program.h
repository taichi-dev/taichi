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
#include "taichi/llvm/llvm_context.h"
#include "taichi/backends/metal/kernel_manager.h"
#include "taichi/backends/opengl/opengl_kernel_launcher.h"
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
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/system/memory_pool.h"
#include "taichi/system/threading.h"
#include "taichi/system/unified_allocator.h"

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

class Program {
 public:
  using Kernel = taichi::lang::Kernel;
  Callable *current_callable{nullptr};
  void *llvm_runtime{nullptr};
  CompileConfig config;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_device{nullptr};
  bool sync{false};  // device/host synchronized?
  bool finalized{false};
  float64 total_compilation_time{0.0};
  static std::atomic<int> num_instances;
  std::unique_ptr<ThreadPool> thread_pool{nullptr};
  std::unique_ptr<MemoryPool> memory_pool{nullptr};
  uint64 *result_buffer{nullptr};  // TODO: move this
  void *preallocated_device_buffer{
      nullptr};  // TODO: move this to memory allocator
  std::unordered_map<int, SNode *> snodes;

  std::unique_ptr<Runtime> runtime_mem_info{nullptr};
  std::unique_ptr<AsyncEngine> async_engine{nullptr};

  std::vector<std::unique_ptr<Kernel>> kernels;
  std::vector<std::unique_ptr<Function>> functions;
  std::unordered_map<FunctionKey, Function *> function_map;

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

  void kernel_profiler_print() {
    profiler->print();
  }

  void kernel_profiler_clear() {
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

  /**
   * Initializes Program#llvm_context_device, if this has not been done.
   *
   * Not thread safe.
   */
  void maybe_initialize_cuda_llvm_context();

  void synchronize();

  // This is more primitive than synchronize(). It directly calls to the
  // targeted GPU backend's synchronization (or commit in Metal's terminology).
  void device_synchronize();
  // See AsyncEngine::flush().
  // Only useful when async mode is enabled.
  void async_flush();

  /**
   * Materializes the runtime.
   */
  void materialize_runtime();

  int get_snode_tree_size();

  void visualize_layout(const std::string &fn);

  struct KernelProxy {
    std::string name;
    Program *prog;
    bool grad;

    Kernel *def(const std::function<void()> &func) {
      return &(prog->kernel(func, name, grad));
    }
  };

  KernelProxy kernel(const std::string &name, bool grad = false) {
    KernelProxy proxy;
    proxy.prog = this;
    proxy.name = name;
    proxy.grad = grad;
    return proxy;
  }

  Kernel &kernel(const std::function<void()> &body,
                 const std::string &name = "",
                 bool grad = false) {
    // Expr::set_allow_store(true);
    auto func = std::make_unique<Kernel>(*this, body, name, grad);
    // Expr::set_allow_store(false);
    kernels.emplace_back(std::move(func));
    return *kernels.back();
  }

  void start_kernel_definition(Kernel *kernel) {
    current_callable = kernel;
  }

  void end_kernel_definition() {
  }

  Function *create_function(const FunctionKey &func_key);

  // TODO: This function is doing two things: 1) compiling CHI IR, and 2)
  // offloading them to each backend. We should probably separate the logic?
  FunctionType compile(Kernel &kernel);

  // Just does the per-backend executable compilation without kernel lowering.
  FunctionType compile_to_backend_executable(Kernel &kernel,
                                             OffloadedStmt *stmt);

  void check_runtime_error();

  // TODO(#2193): Remove get_current_kernel() and get_current_function()?
  inline Kernel &get_current_kernel() const {
    auto *kernel = dynamic_cast<Kernel *>(current_callable);
    TI_ASSERT(kernel);
    return *kernel;
  }

  inline Function *get_current_function() const {
    auto *func = dynamic_cast<Function *>(current_callable);
    return func;
  }

  TaichiLLVMContext *get_llvm_context(Arch arch) {
    if (arch_is_cpu(arch)) {
      return llvm_context_host.get();
    } else {
      return llvm_context_device.get();
    }
  }

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
    return total_compilation_time;
  }

  void finalize();

  static int get_kernel_id() {
    static int id = 0;
    TI_ASSERT(id < 100000);
    return id++;
  }

  static int default_block_dim(const CompileConfig &config);

  void print_list_manager_info(void *list_manager);

  void print_memory_profiler_info();

  template <typename T, typename... Args>
  T runtime_query(const std::string &key, Args... args) {
    TI_ASSERT(arch_uses_llvm(config.arch));

    TaichiLLVMContext *tlctx = nullptr;
    if (llvm_context_device) {
      tlctx = llvm_context_device.get();
    } else {
      tlctx = llvm_context_host.get();
    }

    auto runtime = tlctx->runtime_jit_module;
    runtime->call<void *, Args...>("runtime_" + key, llvm_runtime,
                                   std::forward<Args>(args)...);
    return fetch_result<T>(taichi_result_buffer_runtime_query_id);
  }

  // Returns zero if the SNode is statically allocated
  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  inline SNodeGlobalVarExprMap *get_snode_to_glb_var_exprs() {
    return &snode_to_glb_var_exprs_;
  }

  inline SNodeRwAccessorsBank &get_snode_rw_accessors_bank() {
    return snode_rw_accessors_bank_;
  }

  /**
   * Adds a new SNode tree.
   *
   * @param root The root of the new SNode tree.
   * @return SNode tree ID.
   */
  int add_snode_tree(std::unique_ptr<SNode> root);

  /**
   * Gets the root of a SNode tree.
   *
   * @param tree_id Index of the SNode tree
   * @return Root of the tree
   */
  SNode *get_snode_root(int tree_id);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(Arch arch);

 private:
  /**
   * Materializes a new SNodeTree.
   *
   * JIT compiles the @param tree to backend-specific data types.
   */
  void materialize_snode_tree(SNodeTree *tree);

  /**
   * Sets the attributes of the Exprs that are backed by SNodes.
   */
  void materialize_snode_expr_attributes();

  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void initialize_llvm_runtime_system();

  /**
   * Initializes the SNodes for LLVM based backends.
   */
  void initialize_llvm_runtime_snodes(const SNodeTree *tree,
                                      StructCompiler *scomp);

  std::unique_ptr<llvm::Module> clone_struct_compiler_initial_context(
      TaichiLLVMContext *tlctx);

  // Metal related data structures
  std::optional<metal::CompiledStructs> metal_compiled_structs_;
  std::unique_ptr<metal::KernelManager> metal_kernel_mgr_;
  // OpenGL related data structures
  std::optional<opengl::StructCompiledResult> opengl_struct_compiled_;
  std::unique_ptr<opengl::GLSLLauncher> opengl_kernel_launcher_;
  // SNode information that requires using Program.
  SNodeGlobalVarExprMap snode_to_glb_var_exprs_;
  SNodeRwAccessorsBank snode_rw_accessors_bank_;

  std::vector<std::unique_ptr<SNodeTree>> snode_trees_;

 public:
#ifdef TI_WITH_CC
  // C backend related data structures
  std::unique_ptr<cccp::CCProgram> cc_program;
#endif
};

}  // namespace lang
}  // namespace taichi
