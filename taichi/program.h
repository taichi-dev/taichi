// Program, which is a context for a taichi program execution

#pragma once
#include "ir.h"
#include "kernel.h"
#include "snode.h"
#include "taichi_llvm_context.h"
#include "tlang_util.h"
#include <atomic>
#include <taichi/context.h>
#include <taichi/profiler.h>
#include <taichi/unified_allocator.h>
#if defined(TC_PLATFORM_UNIX)
#include <dlfcn.h>
#endif
#if defined(TC_USE_MPI)
#include <mpi.h>
#endif

TLANG_NAMESPACE_BEGIN

extern Program *current_program;
extern SNode root;

TC_FORCE_INLINE Program &get_current_program() { return *current_program; }

class Program {
public:
  using Kernel = taichi::Tlang::Kernel;
  // Should be copiable
  std::vector<void *> loaded_dlls;
  Kernel *current_kernel;
  SNode *snode_root;
  // pointer to the data structure. assigned to context.buffers[0] during kernel
  // launches
  void *llvm_runtime;
  void *data_structure;
  CompileConfig config;
  CPUProfiler cpu_profiler;
  Context context;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host, llvm_context_device;
  bool sync; // device/host synchronized?
  bool clear_all_gradients_initialized;
  bool finalized;
  float64 total_compilation_time;
  static std::atomic<int> num_instances;

  std::vector<std::unique_ptr<Kernel>> functions;

  std::function<void()> profiler_print_gpu;
  std::function<void()> profiler_clear_gpu;
  std::unique_ptr<ProfilerBase> profiler_llvm;

  std::string layout_fn;

  // mpi variables
  int world_rank;
  int world_size;
  int local_rank;
  int local_size;
  MPI_Comm local_comm;
  int device_id;

  void profiler_print() {
    if (config.use_llvm) {
      profiler_llvm->print();
    } else {
      if (config.arch == Arch::gpu) {
        profiler_print_gpu();
      } else {
        cpu_profiler.print();
      }
    }
  }

  void profiler_clear() {
    if (config.use_llvm) {
      profiler_llvm->clear();
    } else {
      if (config.arch == Arch::gpu) {
        profiler_clear_gpu();
      } else {
        cpu_profiler.clear();
      }
    }
  }

  Context &get_context() {
    context.buffers[0] = data_structure;
    context.cpu_profiler = &cpu_profiler;
    context.runtime = llvm_runtime;
    return context;
  }

  Program() : Program(default_compile_config.arch) {}

  Program(const Program &){
      TC_NOT_IMPLEMENTED // for pybind11..
  }

  Program(Arch arch);

  void initialize_device_llvm_context();

  void synchronize();

  void finalize() {
    current_program = nullptr;
    for (auto &dll : loaded_dlls) {
#if defined(TC_PLATFORM_UNIX)
      dlclose(dll);
#else
      TC_NOT_IMPLEMENTED
#endif
    }
    UnifiedAllocator::free();
    finalized = true;
    num_instances -= 1;

    // finalize mpi comm
    //initialize mpi
  #if defined TC_USE_MPI
    std::cout << "Finializing mpi comm..." << std::endl;
    MPI_Finalize();
  #endif
  }

  ~Program() {
    if (!finalized)
      finalize();
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
                 const std::string &name = "", bool grad = false) {
    // Expr::set_allow_store(true);
    auto func = std::make_unique<Kernel>(*this, body, name, grad);
    // Expr::set_allow_store(false);
    functions.emplace_back(std::move(func));
    return *functions.back();
  }

  void start_function_definition(Kernel *func) { current_kernel = func; }

  void end_function_definition() {}

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

  Kernel &get_snode_reader(SNode *snode);

  Kernel &get_snode_writer(SNode *snode);

  Arch get_host_arch() { return Arch::x86_64; }

  float64 get_total_compilation_time() { return total_compilation_time; }
};

TLANG_NAMESPACE_END
