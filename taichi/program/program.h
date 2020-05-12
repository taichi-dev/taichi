// Program  - Taichi program execution context

#pragma once

#include <functional>
#include <optional>
#include <atomic>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/lang_util.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/backends/metal/kernel_manager.h"
#include "taichi/backends/opengl/opengl_kernel_launcher.h"
#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/program/kernel.h"
#include "taichi/program/profiler.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/runtime/runtime.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/system/memory_pool.h"
#include "taichi/system/threading.h"
#include "taichi/system/unified_allocator.h"

TLANG_NAMESPACE_BEGIN

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

TLANG_NAMESPACE_END

namespace std {
template <>
struct hash<taichi::lang::JITEvaluatorId> {
  std::size_t operator()(taichi::lang::JITEvaluatorId const &id) const
      noexcept {
    return ((std::size_t)id.op | ((std::size_t)id.ret << 8) |
            ((std::size_t)id.lhs << 16) | ((std::size_t)id.rhs << 24) |
            ((std::size_t)id.is_binary << 31)) ^
           (std::hash<std::thread::id>{}(id.thread_id) << 32);
  }
};
}  // namespace std

TLANG_NAMESPACE_BEGIN

extern Program *current_program;

TI_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class StructCompiler;

class AsyncEngine;

class Program {
 public:
  using Kernel = taichi::lang::Kernel;
  Kernel *current_kernel;
  std::unique_ptr<SNode> snode_root;  // pointer to the data structure.
  void *llvm_runtime;
  CompileConfig config;
  Context context;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host, llvm_context_device;
  bool sync;  // device/host synchronized?
  bool finalized;
  float64 total_compilation_time;
  static std::atomic<int> num_instances;
  ThreadPool thread_pool;
  std::unique_ptr<MemoryPool> memory_pool;
  void *result_buffer;               // TODO: move this
  void *preallocated_device_buffer;  // TODO: move this to memory allocator

  std::unique_ptr<Runtime> runtime;
  std::unique_ptr<AsyncEngine> async_engine;

  std::vector<std::unique_ptr<Kernel>> kernels;

  std::unique_ptr<ProfilerBase> profiler;

  std::unordered_map<JITEvaluatorId, std::unique_ptr<Kernel>>
      jit_evaluator_cache;
  std::mutex jit_evaluator_cache_mut;

  Program() : Program(default_compile_config.arch) {
  }

  Program(Arch arch);

  void profiler_print() {
    profiler->print();
  }

  void profiler_clear() {
    profiler->clear();
  }

  void profiler_start(const std::string &name) {
    profiler->start(name);
  }

  void profiler_stop() {
    profiler->stop();
  }

  ProfilerBase *get_profiler() {
    return profiler.get();
  }

  Context &get_context() {
    context.runtime = (LLVMRuntime *)llvm_runtime;
    return context;
  }
  void initialize_device_llvm_context();

  void synchronize();

  void layout(std::function<void()> func) {
    func();
    materialize_layout();
  }

  void visualize_layout(const std::string &fn);

  struct KernelProxy {
    std::string name;
    Program *prog;
    bool grad;

    Kernel &def(const std::function<void()> &func) {
      return prog->kernel(func, name, grad);
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

  void start_function_definition(Kernel *func) {
    current_kernel = func;
  }

  void end_function_definition() {
  }

  FunctionType compile(Kernel &kernel);

  void initialize_runtime_system(StructCompiler *scomp);

  void materialize_layout();

  void check_runtime_error();

  inline Kernel &get_current_kernel() {
    TI_ASSERT(current_kernel);
    return *current_kernel;
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

  void print_snode_tree() {
    snode_root->print();
  }

  void launch_async(Kernel *kernel);

  ~Program();

 private:
  // Metal related data structures
  std::optional<metal::CompiledStructs> metal_compiled_structs_;
  std::unique_ptr<metal::KernelManager> metal_kernel_mgr_;
  // OpenGL related data structures
  std::optional<opengl::StructCompiledResult> opengl_struct_compiled_;
  std::unique_ptr<opengl::GLSLLauncher> opengl_kernel_launcher_;
};

TLANG_NAMESPACE_END
