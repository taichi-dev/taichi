#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_core.h"

static void taichi_sparse_test(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  TiRuntime runtime = ti_create_runtime(arch);

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, aot_mod_ss.str().c_str());

  TiKernel k_fill_img = ti_get_aot_module_kernel(aot_mod, "fill_img");
  TiKernel k_block1_deactivate_all =
      ti_get_aot_module_kernel(aot_mod, "block1_deactivate_all");
  TiKernel k_activate = ti_get_aot_module_kernel(aot_mod, "activate");
  TiKernel k_paint = ti_get_aot_module_kernel(aot_mod, "paint");
  TiKernel k_check_img_value =
      ti_get_aot_module_kernel(aot_mod, "check_img_value");

  constexpr uint32_t arg_count = 1;
  TiArgument args[arg_count];

  ti_launch_kernel(runtime, k_fill_img, 0, &args[0]);
  for (int i = 0; i < 100; i++) {
    float val = 0.05f * i;
    TiArgument base_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_F32,
                           .value = {.f32 = val}};
    args[0] = std::move(base_arg);

    ti_launch_kernel(runtime, k_block1_deactivate_all, 0, &args[0]);
    ti_launch_kernel(runtime, k_activate, arg_count, &args[0]);
    ti_launch_kernel(runtime, k_paint, 0, &args[0]);
  }

  // Accuracy Check
  ti_launch_kernel(runtime, k_check_img_value, 0, &args[0]);

  // Check Results
  capi::utils::check_runtime_error(runtime);

  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

TEST(CapiTaichiSparseTest, Cuda) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    taichi_sparse_test(arch);
  }
}
