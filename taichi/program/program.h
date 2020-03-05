// Program class - a context for a Taichi program execution

#pragma once

#include <optional>
#include <atomic>

#define TI_RUNTIME_HOST
#include "taichi/ir/ir.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/snode.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/lang_util.h"
#include "taichi/runtime/llvm/context.h"
#include "taichi/program/profiler.h"
#include "taichi/system/threading.h"
#include "taichi/system/unified_allocator.h"
#include "taichi/system/memory_pool.h"
#include "taichi/platform/metal/metal_kernel_util.h"
#include "taichi/platform/metal/metal_runtime.h"
#include "taichi/platform/opengl/opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

extern Program *current_program;

TI_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class StructCompiler;

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

  std::vector<std::unique_ptr<Kernel>> functions;

  std::unique_ptr<ProfilerBase> profiler;

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
    context.runtime = (Runtime *)llvm_runtime;
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
    functions.emplace_back(std::move(func));
    return *functions.back();
  }

  void start_function_definition(Kernel *func) {
    current_kernel = func;
  }

  void end_function_definition() {
  }

  FunctionType compile(Kernel &kernel);

  void initialize_runtime_system(StructCompiler *scomp);

  void materialize_layout();

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

  ~Program();

 private:
  std::optional<metal::StructCompiledResult> metal_struct_compiled_;
  std::optional<opengl::StructCompiledResult> opengl_struct_compiled_;
  std::unique_ptr<metal::MetalRuntime> metal_runtime_;
};

TLANG_NAMESPACE_END
