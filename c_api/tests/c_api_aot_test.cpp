#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"

static void kernel_aot_test(TiArch arch) {
  uint32_t kArrLen = 32;
  int arg0_val = 0;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::NdArray<int32_t> arg1_array =
      runtime.allocate_ndarray<int32_t>({kArrLen}, {1}, true);
  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::Kernel k_run = aot_mod.get_kernel("run");

  k_run[0] = arg0_val;
  k_run[1] = arg1_array;
  k_run.launch();
  runtime.wait();

  // Check Results
  int32_t *data = reinterpret_cast<int32_t *>(arg1_array.map());

  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i);
  }

  arg1_array.unmap();
}

static void field_aot_test(TiArch arch) {
  int base_val = 10;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);
  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());

  ti::Kernel k_init_fields = aot_mod.get_kernel("init_fields");
  ti::Kernel k_check_init_x = aot_mod.get_kernel("check_init_x");
  ti::Kernel k_check_init_y = aot_mod.get_kernel("check_init_y");
  ti::Kernel k_deactivate_pointer_fields =
      aot_mod.get_kernel("deactivate_pointer_fields");
  ti::Kernel k_activate_pointer_fields =
      aot_mod.get_kernel("activate_pointer_fields");
  ti::Kernel k_check_deactivate_pointer_fields =
      aot_mod.get_kernel("check_deactivate_pointer_fields");
  ti::Kernel k_check_activate_pointer_fields =
      aot_mod.get_kernel("check_activate_pointer_fields");

  k_init_fields[0] = base_val;
  k_init_fields.launch();
  k_check_init_x[0] = base_val;
  k_check_init_x.launch();
  k_check_init_y.launch();
  k_deactivate_pointer_fields.launch();
  k_check_deactivate_pointer_fields.launch();
  k_activate_pointer_fields.launch();
  k_check_activate_pointer_fields.launch();
  runtime.wait();

  // Check Results
  capi::utils::check_runtime_error(runtime);
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

TEST(CapiAotTest, CpuKernel) {
  TiArch arch = TiArch::TI_ARCH_X64;
  kernel_aot_test(arch);
}

TEST(CapiAotTest, CudaKernel) {
  if (capi::utils::is_cuda_available()) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    kernel_aot_test(arch);
  }
}

TEST(CapiAotTest, VulkanKernel) {
  if (capi::utils::is_vulkan_available()) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    kernel_aot_test(arch);
  }
}

TEST(CapiAotTest, OpenglKernel) {
  if (capi::utils::is_opengl_available()) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    kernel_aot_test(arch);
  }
}
