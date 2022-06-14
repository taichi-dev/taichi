#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/system/memory_pool.h"
#include "taichi/backends/cpu/aot_module_loader_impl.h"
#include "taichi/backends/cuda/aot_module_loader_impl.h"
#include "taichi/llvm/llvm_aot_module_loader.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

TEST(LlvmAOTTest, Field) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmProgramImpl prog{cfg, kNoProfiler};
  auto *compute_device = prog.get_compute_device();

  // Must have handled all the arch fallback logic by this point.
  auto memory_pool = std::make_unique<MemoryPool>(cfg.arch, compute_device);
  prog.initialize_host();
  uint64 *result_buffer{nullptr};
  prog.materialize_runtime(memory_pool.get(), kNoProfiler, &result_buffer);

  cpu::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.program = &prog;
  std::unique_ptr<aot::Module> mod = cpu::make_aot_module(aot_params);

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

  // Initialize Fields
  aot::Field *field_x = mod->get_field("0" /*snode_tree_id*/);
  aot::Field *field_y = mod->get_field("0" /*snode_tree_id*/);

  finalize_aot_field(mod.get(), field_x, result_buffer);
  finalize_aot_field(mod.get(), field_y, result_buffer);

  int base_value = 10;
  /* -------- Test Case 1 ------ */
  // Kernel: init_fields(int)
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    ctx.set_arg(0, base_value);
    k_init_fields->launch(&ctx);
  }

  // Kernel: check_init_x(int)
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    ctx.set_arg(0, base_value);
    k_check_init_x->launch(&ctx);
  }
  // Kernel: check_init_y()
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    k_check_init_y->launch(&ctx);
  }

  /* -------- Test Case 2 ------ */
  // Kernel: deactivate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    k_deactivate_pointer_fields->launch(&ctx);
  }
  // Kernel: check_deactivate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    k_check_deactivate_pointer_fields->launch(&ctx);
  }

  /* -------- Test Case 3 ------ */
  // Kernel: activate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    k_activate_pointer_fields->launch(&ctx);
  }
  // Kernel: check_activate_pointer_fields()
  {
    RuntimeContext ctx;
    ctx.runtime = prog.get_llvm_runtime();
    k_check_activate_pointer_fields->launch(&ctx);
  }

  // Check assertion error from ti.kernel
  prog.check_runtime_error(result_buffer);
}

}  // namespace lang
}  // namespace taichi
