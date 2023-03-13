#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

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

  std::vector<int> arg2_v = {1, 2, 3};

  // This is just to make sure clear_args() does its work.
  k_run.push_arg(arg0_val);
  k_run.clear_args();

  k_run.push_arg(arg0_val);
  k_run.push_arg(arg1_array);
  k_run.push_arg(arg2_v);
  k_run.launch();
  runtime.wait();

  // Check Results
  int32_t *data = reinterpret_cast<int32_t *>(arg1_array.map());

  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i + arg0_val + arg2_v[0]);
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

void texture_aot_kernel_test(TiArch arch) {
  const uint32_t width = 128;
  const uint32_t height = 128;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str());

  ti::Kernel k_run0 = aot_mod.get_kernel("run0");
  ti::Kernel k_run1 = aot_mod.get_kernel("run1");
  ti::Kernel k_run2 = aot_mod.get_kernel("run2");

  ti::Texture tex0 =
      runtime.allocate_texture2d(width, height, TI_FORMAT_R32F, TI_NULL_HANDLE);
  ti::Texture tex1 =
      runtime.allocate_texture2d(width, height, TI_FORMAT_R32F, TI_NULL_HANDLE);
  ti::NdArray<float> arr =
      runtime.allocate_ndarray<float>({width, height}, {}, true);

  k_run0[0] = tex0;
  k_run1[0] = tex0;
  k_run1[1] = tex1;
  k_run2[0] = tex0;
  k_run2[1] = tex1;
  k_run2[2] = arr;

  k_run0.launch();
  k_run1.launch();
  k_run2.launch();
  runtime.wait();

  std::vector<float> arr_data(128 * 128);
  arr.read(arr_data);
  for (auto x : arr_data) {
    EXPECT_GT(x, 0.5);
  }
}

static void shared_array_aot_test(TiArch arch) {
  uint32_t kArrLen = 8192;

  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::NdArray<float> v_array =
      runtime.allocate_ndarray<float>({kArrLen}, {}, true);
  ti::NdArray<float> d_array =
      runtime.allocate_ndarray<float>({kArrLen}, {}, true);
  ti::NdArray<float> a_array =
      runtime.allocate_ndarray<float>({kArrLen}, {}, true);
  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::Kernel k_run = aot_mod.get_kernel("run");

  k_run.push_arg(v_array);
  k_run.push_arg(d_array);
  k_run.push_arg(a_array);
  k_run.launch();
  runtime.wait();

  // Check Results
  float *data = reinterpret_cast<float *>(a_array.map());

  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], kArrLen);
  }

  a_array.unmap();
}

TEST_F(CapiTest, AotTestCpuField) {
  TiArch arch = TiArch::TI_ARCH_X64;
  field_aot_test(arch);
}

TEST_F(CapiTest, AotTestCudaField) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    field_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestCpuKernel) {
  TiArch arch = TiArch::TI_ARCH_X64;
  kernel_aot_test(arch);
}

TEST_F(CapiTest, AotTestCudaKernel) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    kernel_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestVulkanKernel) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    kernel_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestMetalKernel) {
  if (ti::is_arch_available(TI_ARCH_METAL)) {
    TiArch arch = TiArch::TI_ARCH_METAL;
    kernel_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestOpenglKernel) {
  if (ti::is_arch_available(TI_ARCH_OPENGL)) {
    TiArch arch = TiArch::TI_ARCH_OPENGL;
    kernel_aot_test(arch);
  }
}

TEST_F(CapiTest, GraphTestVulkanTextureKernel) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    texture_aot_kernel_test(arch);
  }
}

TEST_F(CapiTest, GraphTestMetalTextureKernel) {
  if (ti::is_arch_available(TI_ARCH_METAL)) {
    TiArch arch = TiArch::TI_ARCH_METAL;
    texture_aot_kernel_test(arch);
  }
}

TEST_F(CapiTest, AotTestCudaSharedArray) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    shared_array_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestVulkanSharedArray) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    shared_array_aot_test(arch);
  }
}

TEST_F(CapiTest, AotTestMetalSharedArray) {
  if (ti::is_arch_available(TI_ARCH_METAL)) {
    TiArch arch = TiArch::TI_ARCH_METAL;
    shared_array_aot_test(arch);
  }
}
