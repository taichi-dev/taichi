// Program class - a context for a Taichi program execution

#pragma once

#define TI_RUNTIME_HOST
#include "ir.h"
#include "kernel.h"
#include "snode.h"
#include "taichi_llvm_context.h"
#include "tlang_util.h"
#include <atomic>
#include <taichi/context.h>
#include <taichi/profiler.h>
#include <taichi/system/threading.h>
#include <taichi/unified_allocator.h>
#include "memory_pool.h"
#include <optional>
#include <taichi/platform/metal/metal_kernel_util.h>
#include <taichi/platform/metal/metal_runtime.h>

#if defined(TI_PLATFORM_UNIX)
#include <dlfcn.h>
#endif

TLANG_NAMESPACE_BEGIN

extern Program *current_program;

TI_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class Program {
 public:
  using Kernel = taichi::Tlang::Kernel;
  // Should be copiable
  std::vector<void *> loaded_dlls;
  Kernel *current_kernel;
  std::unique_ptr<SNode> snode_root;
  // pointer to the data structure. assigned to context.buffers[0] during kernel
  // launches
  void *llvm_runtime;
  CompileConfig config;
  ProfilerBase *cpu_profiler;
  Context context;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host, llvm_context_device;
  bool sync;  // device/host synchronized?
  bool finalized;
  float64 total_compilation_time;
  static std::atomic<int> num_instances;
  ThreadPool thread_pool;
  std::unique_ptr<MemoryPool> memory_pool;

  std::vector<std::unique_ptr<Kernel>> functions;

  std::function<void()> profiler_print_gpu;
  std::function<void()> profiler_clear_gpu;
  std::unique_ptr<ProfilerBase> profiler_llvm;

  std::string layout_fn;

  Program() : Program(default_compile_config.arch) {
  }

  Program(Arch arch);

  void profiler_print() {
    if (config.use_llvm) {
      profiler_llvm->print();
    } else {
      if (config.arch == Arch::cuda) {
        profiler_print_gpu();
      } else {
        cpu_profiler->print();
      }
    }
  }

  void profiler_clear() {
    if (config.use_llvm) {
      profiler_llvm->clear();
    } else {
      if (config.arch == Arch::cuda) {
        profiler_clear_gpu();
      } else {
        cpu_profiler->clear();
      }
    }
  }

  void profiler_start(const std::string &name) {
    TI_ASSERT(config.use_llvm);
    profiler_llvm->start(name);
  }

  void profiler_stop() {
    TI_ASSERT(config.use_llvm);
    profiler_llvm->stop();
  }

  ProfilerBase *get_profiler() {
    TI_ASSERT(config.use_llvm);
    return profiler_llvm.get();
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

  void materialize_layout();

  inline Kernel &get_current_kernel() {
    TI_ASSERT(current_kernel);
    return *current_kernel;
  }

  TaichiLLVMContext *get_llvm_context(Arch arch) {
    if (arch == Arch::x64) {
      return llvm_context_host.get();
    } else {
      return llvm_context_device.get();
    }
  }

  Kernel &get_snode_reader(SNode *snode);

  Kernel &get_snode_writer(SNode *snode);

  Arch get_host_arch() {
    return Arch::x64;
  }

  float64 get_total_compilation_time() {
    return total_compilation_time;
  }

  void finalize();

  ~Program();

 private:
  std::optional<metal::StructCompiledResult> metal_struct_compiled_;
  std::unique_ptr<metal::MetalRuntime> metal_runtime_;
};

TLANG_NAMESPACE_END
