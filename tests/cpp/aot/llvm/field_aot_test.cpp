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

static void run_field_tests(aot::Module *mod,
                            LlvmRuntimeExecutor *exec,
                            uint64 *result_buffer) {
  aot::Kernel *k_init_fields = mod->get_kernel("init_fields");
  aot::Kernel *k_check_init_x = mod->get_kernel("check_init_x");
  aot::Kernel *k_check_init_y = mod->get_kernel("check_init_y");

  aot::Kernel *k_deactivate_pointer_fields =
      mod->get_kernel("deactivate_pointer_fields");
  aot::Kernel *k_activate_pointer_fields =
      mod->get_kernel("activate_pointer_fields");

  aot::Kernel *k_check_deactivate_pointer_fields =
      mod->get_kernel("check_deactivate_pointer_fields");
  aot::Kernel *k_check_activate_pointer_fields =
      mod->get_kernel("check_activate_pointer_fields");

  // Initialize SNodeTree
  aot::Field *snode_tree_0 = mod->get_snode_tree("0" /*snode_tree_id*/);
  allocate_aot_snode_tree_type(mod, snode_tree_0, result_buffer);

  int base_value = 10;
  /* -------- Test Case 1 ------ */
  // Kernel: init_fields(int)
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    ctx.set_arg(0, base_value);
    k_init_fields->launch(&ctx);
  }

  // Kernel: check_init_x(int)
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    ctx.set_arg(0, base_value);
    k_check_init_x->launch(&ctx);
  }
  // Kernel: check_init_y()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_check_init_y->launch(&ctx);
  }

  /* -------- Test Case 2 ------ */
  // Kernel: deactivate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_deactivate_pointer_fields->launch(&ctx);
  }
  // Kernel: check_deactivate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_check_deactivate_pointer_fields->launch(&ctx);
  }

  /* -------- Test Case 3 ------ */
  // Kernel: activate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_activate_pointer_fields->launch(&ctx);
  }
  // Kernel: check_activate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = exec->get_llvm_runtime();
    k_check_activate_pointer_fields->launch(&ctx);
  }

  // Check assertion error from ti.kernel
  exec->check_runtime_error(result_buffer);
}

TEST(LlvmAotTest, CpuField) {
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

  run_field_tests(mod.get(), &exec, result_buffer);
}

TEST(LlvmAotTest, CudaField) {
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

    run_field_tests(mod.get(), &exec, result_buffer);
  }
}

}  // namespace lang
}  // namespace taichi
