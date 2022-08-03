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

namespace taichi {
namespace lang {

static void run_bitmasked_tests(aot::Module *mod,
                                LlvmRuntimeExecutor *exec,
                                uint64 *result_buffer) {
  aot::Kernel *k_activate = mod->get_kernel("activate");
  aot::Kernel *k_check_value_0 = mod->get_kernel("check_value_0");
  aot::Kernel *k_deactivate = mod->get_kernel("deactivate");
  aot::Kernel *k_check_value_1 = mod->get_kernel("check_value_1");

  // Initialize SNodeTree
  aot::Field *snode_tree_0 = mod->get_snode_tree("0" /*snode_tree_id*/);
  allocate_aot_snode_tree_type(mod, snode_tree_0, result_buffer);

  /* -------- Test Case 1 ------ */
  // Kernel: activate()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_activate->launch(&ctx);
  }

  // Kernel: check_value_0()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_check_value_0->launch(&ctx);
  }

  /* -------- Test Case 2 ------ */
  // Kernel: deactivate()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_deactivate->launch(&ctx);
  }
  // Kernel: check_value_1()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_check_value_1->launch(&ctx);
  }

  // Check assertion error from ti.kernel
  exec->check_runtime_error(result_buffer);
}

TEST(LlvmAotTest, CpuBitmasked) {
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

  run_bitmasked_tests(mod.get(), &exec, result_buffer);
}

TEST(LlvmAotTest, CudaBitmasked) {
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

    run_bitmasked_tests(mod.get(), &exec, result_buffer);
  }
}

}  // namespace lang
}  // namespace taichi
