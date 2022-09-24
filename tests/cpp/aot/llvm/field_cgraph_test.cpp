#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/system/memory_pool.h"
#include "taichi/runtime/cpu/aot_module_loader_impl.h"
#include "taichi/runtime/cuda/aot_module_loader_impl.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

void run_graph_tests(aot::Module *mod,
                     LlvmRuntimeExecutor *exec,
                     uint64 *result_buffer) {
  // Initialize SNodeTree
  aot::Field *snode_tree_0 = mod->get_snode_tree("0" /*snode_tree_id*/);
  allocate_aot_snode_tree_type(mod, snode_tree_0, result_buffer);

  /* Test with Graph */
  // Prepare & Run "init" Graph
  auto run_graph = mod->get_graph("run_graph");

  int base0 = 10;
  int base1 = 10;
  std::unordered_map<std::string, taichi::lang::aot::IValue> args;
  args.insert({"base0", taichi::lang::aot::IValue::create(base0)});
  args.insert({"base1", taichi::lang::aot::IValue::create(base1)});

  run_graph->run(args);

  // Check assertion error from ti.kernel
  exec->check_runtime_error(result_buffer);
}

TEST(LlvmCGraph, CpuField) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};
  auto *compute_device = exec.get_compute_device();

  // Must have handled all the arch fallback logic by this point.
  auto memory_pool = std::make_unique<MemoryPool>(cfg.arch, compute_device);
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(memory_pool.get(), kNoProfiler, &result_buffer);

  cpu::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.executor_ = &exec;
  std::unique_ptr<aot::Module> mod = cpu::make_aot_module(aot_params);

  run_graph_tests(mod.get(), &exec, result_buffer);
}

TEST(LlvmCGraph, CudaField) {
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    uint64 *result_buffer{nullptr};
    exec.materialize_runtime(nullptr, kNoProfiler, &result_buffer);

    cuda::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    auto mod = cuda::make_aot_module(aot_params);

    run_graph_tests(mod.get(), &exec, result_buffer);
  }
}

}  // namespace taichi::lang
