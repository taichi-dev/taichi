// Program, which is a context for a taichi program execution

#pragma once

#include "../include/context.h"
#include "../include/unified_allocator.h"
#include "../include/profiler.h"
#include "util.h"
#include "snode.h"
#include "ir.h"
#include "taichi_llvm_context.h"
#include "kernel.h"
#include <dlfcn.h>

TLANG_NAMESPACE_BEGIN

extern Program *current_program;
extern SNode root;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class Program {
public:
  using Kernel = taichi::Tlang::Kernel;
  // Should be copiable
  std::vector<void *> loaded_dlls;
  Kernel *current_kernel;
  SNode *current_snode;
  SNode *snode_root;
  // pointer to the data structure. assigned to context.buffers[0] during kernel
  // launches
  void *llvm_runtime;
  void *data_structure;
  CompileConfig config;
  CPUProfiler cpu_profiler;
  Context context;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host, llvm_context_device;
  bool sync;  // device/host synchronized?
  bool clear_all_gradients_initialized;

  std::vector<std::unique_ptr<Kernel>> functions;
  int index_counter;

  std::function<void()> profiler_print_gpu;
  std::function<void()> profiler_clear_gpu;

  std::string layout_fn;

  void profiler_print() {
    if (config.arch == Arch::gpu) {
      profiler_print_gpu();
    } else {
      cpu_profiler.print();
    }
  }

  void profiler_clear() {
    if (config.arch == Arch::gpu) {
      profiler_clear_gpu();
    } else {
      cpu_profiler.clear();
    }
  }

  Context get_context() {
    context.buffers[0] = data_structure;
    context.cpu_profiler = &cpu_profiler;
    context.runtime = llvm_runtime;
    return context;
  }

  Program() : Program(default_compile_config.arch) {
  }

  Program(const Program &) {
    TC_NOT_IMPLEMENTED  // for pybind11..
  }

  Program(Arch arch);

  void initialize_device_llvm_context();

  void synchronize();

  ~Program() {
    current_program = nullptr;
    for (auto &dll : loaded_dlls) {
      dlclose(dll);
    }
    UnifiedAllocator::free();
  }

  void layout(std::function<void()> func) {
    root = SNode(0, SNodeType::root);
    snode_root = &root;
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
    current_snode = nullptr;
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
    TC_ASSERT(current_kernel);
    return *current_kernel;
  }

  TaichiLLVMContext *get_llvm_context(Arch arch) {
    if (arch == Arch::x86_64) {
      return llvm_context_host.get();
    } else {
      return llvm_context_device.get();
    }
  }

  std::vector<std::function<void()>> gradient_clearers;

  void initialize_gradient_clearers();

  void clear_all_gradients();
};

TLANG_NAMESPACE_END
