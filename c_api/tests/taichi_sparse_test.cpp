#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

static void taichi_sparse_test(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  // Load Aot and Kernel
  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());

  ti::Kernel k_fill_img = aot_mod.get_kernel("fill_img");
  ti::Kernel k_block1_deactivate_all =
      aot_mod.get_kernel("block1_deactivate_all");
  ti::Kernel k_activate = aot_mod.get_kernel("activate");
  ti::Kernel k_paint = aot_mod.get_kernel("paint");
  ti::Kernel k_check_img_value = aot_mod.get_kernel("check_img_value");

  k_fill_img.launch();
  float val = 0.05f;

  k_block1_deactivate_all.launch();
  k_activate[0] = val;
  k_activate.launch();
  k_paint.launch();

  // Accuracy Check
  k_check_img_value.launch();
  runtime.wait();

  // Check Results
  capi::utils::check_runtime_error(runtime);
}

TEST_F(CapiTest, TaichiSparseTestCuda) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    taichi_sparse_test(arch);
  }
}
