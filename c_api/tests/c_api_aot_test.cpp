#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

void field_aot_test(TiArch arch) {
  int base_val = 10;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, aot_mod_ss.str().c_str());

  TiKernel k_init_fields = ti_get_aot_module_kernel(aot_mod, "init_fields");
  TiKernel k_check_init_x = ti_get_aot_module_kernel(aot_mod, "check_init_x");
  TiKernel k_check_init_y = ti_get_aot_module_kernel(aot_mod, "check_init_y");
  TiKernel k_deactivate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "deactivate_pointer_fields");
  TiKernel k_activate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "activate_pointer_fields");
  TiKernel k_check_deactivate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "check_deactivate_pointer_fields");
  TiKernel k_check_activate_pointer_fields =
      ti_get_aot_module_kernel(aot_mod, "check_activate_pointer_fields");

  // Prepare Arguments
  TiArgument base_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                         .value = {.i32 = base_val}};

  constexpr uint32_t arg_count = 1;
  TiArgument args[arg_count] = {std::move(base_arg)};

  // Kernel Execution
  ti_launch_kernel(runtime, k_init_fields, arg_count, &args[0]);
  ti_launch_kernel(runtime, k_check_init_x, arg_count, &args[0]);
  ti_launch_kernel(runtime, k_check_init_y, 0 /*arg_count*/, &args[0]);

  ti_launch_kernel(runtime, k_deactivate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_check_deactivate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_activate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);
  ti_launch_kernel(runtime, k_check_activate_pointer_fields, 0 /*arg_count*/,
                   &args[0]);

  // Check Results
  capi::utils::check_runtime_error(runtime);

  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiAotTest, CpuField) {
  TiArch arch = TiArch::TI_ARCH_X64;
  field_aot_test(arch);
}

TEST(CapiAotTest, CudaField) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    field_aot_test(arch);
  }
}
