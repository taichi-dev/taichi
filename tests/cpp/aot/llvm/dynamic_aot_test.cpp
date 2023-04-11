#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/cpu/kernel_launcher.h"

#ifdef TI_WITH_CUDA

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

#endif

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

static void run_dynamic_tests(aot::Module *mod,
                              LlvmRuntimeExecutor *exec,
                              uint64 *result_buffer) {
  aot::Kernel *k_activate = mod->get_kernel("activate");
  aot::Kernel *k_check_value_0 = mod->get_kernel("check_value_0");
  aot::Kernel *k_deactivate = mod->get_kernel("deactivate");
  aot::Kernel *k_check_value_1 = mod->get_kernel("check_value_1");

  // Initialize SNodeTree
  aot::Field *snode_tree_0 = mod->get_snode_tree("0" /*snode_tree_id*/);
  LLVM::allocate_aot_snode_tree_type(mod, snode_tree_0, result_buffer);

  /* -------- Test Case 1 ------ */
  // Kernel: activate()
  {
    LaunchContextBuilder builder(k_activate);
    k_activate->launch(builder);
  }

  // Kernel: check_value_0()
  {
    LaunchContextBuilder builder(k_check_value_0);
    k_check_value_0->launch(builder);
  }

  /* -------- Test Case 2 ------ */
  // Kernel: deactivate()
  {
    LaunchContextBuilder builder(k_deactivate);
    k_deactivate->launch(builder);
  }
  // Kernel: check_value_1()
  {
    LaunchContextBuilder builder(k_check_value_1);
    k_check_value_1->launch(builder);
  }

  // Check assertion error from ti.kernel
  exec->check_runtime_error(result_buffer);
}

TEST(LlvmAotTest, CpuDynamic) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};

  // Must have handled all the arch fallback logic by this point.
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(kNoProfiler, &result_buffer);

  LLVM::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.executor_ = &exec;
  aot_params.kernel_launcher =
      std::make_unique<cpu::KernelLauncher>(cpu::KernelLauncher::Config{&exec});
  std::unique_ptr<aot::Module> mod =
      LLVM::make_aot_module(std::move(aot_params));

  run_dynamic_tests(mod.get(), &exec, result_buffer);
}

TEST(LlvmAotTest, CudaDynamic) {
#ifdef TI_WITH_CUDA
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    uint64 *result_buffer{nullptr};
    exec.materialize_runtime(kNoProfiler, &result_buffer);

    LLVM::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    aot_params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
        cuda::KernelLauncher::Config{&exec});
    std::unique_ptr<aot::Module> mod =
        LLVM::make_aot_module(std::move(aot_params));

    run_dynamic_tests(mod.get(), &exec, result_buffer);
  }
#endif
}

}  // namespace taichi::lang
