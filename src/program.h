#pragma once

#include "../include/context.h"
#include "../include/unified_allocator.h"
#include "../include/profiler.h"
#include "util.h"
#include "snode.h"
#include "ir.h"
#include "taichi_llvm_context.h"
#include <dlfcn.h>

TLANG_NAMESPACE_BEGIN

class Program;
extern Program *current_program;
extern SNode root;

TC_FORCE_INLINE Program &get_current_program() {
  return *current_program;
}

class Kernel {
 public:
  std::unique_ptr<IRNode> ir_holder;
  IRNode *ir;
  Program &program;
  FunctionType compiled;
  std::string name;
  std::vector<DataType> args;
  bool benchmarking;
  bool is_reduction;  // TODO: systematically treat all types of reduction
  bool grad;

  Kernel(Program &program,
         std::function<void()> func,
         std::string name = "",
         bool grad = false);

  void compile();

  void operator()();

  std::function<void()> func() {
    return std::function<void()>([&] { (*this)(); });
  }

  int insert_arg(DataType dt) {
    args.push_back(dt);
    return args.size() - 1;
  }

  void set_arg_float(int i, float64 d);

  void set_arg_int(int i, int64 d);
};

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
  TaichiLLVMContext llvm_context;
  bool sync;  // device/host synchronized?

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
    return context;
  }

  Program() : Program(default_compile_config.arch) {
  }

  Program(const Program &){
      TC_NOT_IMPLEMENTED  // for pybind11..
  }

  Program(Arch arch) {
#if !defined(CUDA_FOUND)
    if (arch == Arch::gpu) {
      TC_WARN("CUDA not found. GPU is not supported.");
      TC_WARN("Falling back to x86_64");
      arch = Arch::x86_64;
    }
#endif
    UnifiedAllocator::create();
    TC_ASSERT(current_program == nullptr);
    current_program = this;
    config = default_compile_config;
    config.arch = arch;
    current_kernel = nullptr;
    snode_root = nullptr;
    index_counter = 0;
    sync = true;
    llvm_runtime = nullptr;
  }

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
};

TLANG_NAMESPACE_END
