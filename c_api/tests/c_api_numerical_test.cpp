#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

#ifdef TI_WITH_VULKAN

std::vector<float> read_fp16_ndarray(ti::Runtime &runtime,
                                     const ti::NdArray<uint16_t> &ndarray,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<uint32_t> &elem_shape) {
  auto tmp = runtime.allocate_ndarray<uint16_t>(
      ndarray.elem_type(), shape, elem_shape, /*host_accessible=*/true);
  ndarray.slice().copy_to(tmp.slice());
  runtime.wait();

  std::vector<uint16_t> data(tmp.scalar_count());
  tmp.read(data);

  std::vector<float> res(tmp.scalar_count());
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = capi::utils::to_float32(data[i]);
  }

  tmp.destroy();
  return res;
}

TEST_F(CapiTest, Float16Fill) {
  TiArch arch = TiArch::TI_ARCH_VULKAN;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  std::vector<const char *> device_extensions = {
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME};
  TiRuntime ti_runtime = ti_create_vulkan_runtime_ext(
      VK_API_VERSION_1_0, 0, {} /*instance extensions*/, 1,
      device_extensions.data() /*device extensions*/);
  ti::Runtime runtime = ti::Runtime(arch, ti_runtime, true);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::Kernel k_fill_scalar_array_with_fp32 =
      aot_mod.get_kernel("fill_scalar_array_with_fp32");
  ti::Kernel k_fill_scalar_array_with_fp16 =
      aot_mod.get_kernel("fill_scalar_array_with_fp16");
  ti::Kernel k_fill_matrix_array_with_fp16 =
      aot_mod.get_kernel("fill_matrix_array_with_fp16");

  uint32_t kArrLen = 32;
  std::vector<uint32_t> elem_shape = {2, 3};
  ti::NdArray<uint16_t> scalar_array = runtime.allocate_ndarray<uint16_t>(
      TI_DATA_TYPE_F16, {kArrLen}, {}, false);
  ti::NdArray<uint16_t> matrix_array = runtime.allocate_ndarray<uint16_t>(
      TI_DATA_TYPE_F16, {kArrLen}, elem_shape, false);
  /* -------------------- */
  /* fill_scalar_array_with_fp32 */
  /* -------------------- */
  float fill_fp32_val = -1.0;

  k_fill_scalar_array_with_fp32[0] = scalar_array;
  k_fill_scalar_array_with_fp32[1] = fill_fp32_val;
  k_fill_scalar_array_with_fp32.launch();
  runtime.wait();

  // Check Results
  {
    std::vector<float> data =
        read_fp16_ndarray(runtime, scalar_array, {kArrLen}, {});
    for (float x : data) {
      EXPECT_EQ(x, -1.0);
    }
  }

  /* -------------------- */
  /* fill_scalar_array_with_fp16 */
  /* -------------------- */
  float fill_fp16_val = -5.0;

  k_fill_scalar_array_with_fp16[0] = scalar_array;
  k_fill_scalar_array_with_fp16[1].set_f16(fill_fp16_val);
  k_fill_scalar_array_with_fp16.launch();
  runtime.wait();

  // Check Results
  {
    std::vector<float> data =
        read_fp16_ndarray(runtime, scalar_array, {kArrLen}, {});
    for (float x : data) {
      EXPECT_EQ(x, -5.0);
    }
  }

  /* -------------------- */
  /* fill_matrix_array_with_fp16 */
  /* -------------------- */
  float fill_fp16_val2 = float(8.0);

  k_fill_matrix_array_with_fp16[0] = matrix_array;
  k_fill_matrix_array_with_fp16[1].set_f16(fill_fp16_val2);
  k_fill_matrix_array_with_fp16.launch();
  runtime.wait();

  // Check Results
  {
    std::vector<float> data =
        read_fp16_ndarray(runtime, matrix_array, {kArrLen}, elem_shape);
    for (float x : data) {
      EXPECT_EQ(x, 8.0);
    }
  }
}

TEST_F(CapiTest, Float16Compute) {
  TiArch arch = TiArch::TI_ARCH_VULKAN;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  std::vector<const char *> device_extensions = {
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME};
  TiRuntime ti_runtime = ti_create_vulkan_runtime_ext(
      VK_API_VERSION_1_0, 0, {} /*instance extensions*/, 1,
      device_extensions.data() /*device extensions*/);
  ti::Runtime runtime = ti::Runtime(arch, ti_runtime, true);

  ti::AotModule aot_mod = runtime.load_aot_module(aot_mod_ss.str().c_str());
  ti::Kernel k_compute = aot_mod.get_kernel("compute_kernel");

  uint32_t kArrLen = 32;
  std::vector<uint32_t> elem_shape = {2, 3};
  ti::NdArray<uint16_t> pose =
      runtime.allocate_ndarray<uint16_t>(TI_DATA_TYPE_F16, {}, {3, 4}, true);
  ti::NdArray<uint16_t> direction = runtime.allocate_ndarray<uint16_t>(
      TI_DATA_TYPE_F16, {kArrLen}, {1, 3}, true);
  ti::NdArray<uint16_t> out_0 = runtime.allocate_ndarray<uint16_t>(
      TI_DATA_TYPE_F16, {kArrLen}, {3}, false);
  ti::NdArray<uint16_t> out_1 = runtime.allocate_ndarray<uint16_t>(
      TI_DATA_TYPE_F16, {kArrLen}, {3}, false);

  // Fill pose with "0.3"

  std::vector<uint16_t> pose_val_fp16(3 * 4);
  for (size_t i = 0; i < pose_val_fp16.size(); i++) {
    pose_val_fp16[i] = capi::utils::to_float16(static_cast<float>(i));
  }
  pose.write(pose_val_fp16);

  std::vector<uint16_t> direction_val_fp16(kArrLen * 1 * 3);
  for (size_t i = 0; i < direction_val_fp16.size(); i++) {
    direction_val_fp16[i] = capi::utils::to_float16(static_cast<float>(i));
  }
  direction.write(pose_val_fp16);

  /* -------------------- */
  /* fill_scalar_array_with_fp16 */
  /* -------------------- */
  k_compute[0] = pose;
  k_compute[1] = direction;
  k_compute[2] = out_0;
  k_compute[3] = out_1;
  k_compute.launch();
  runtime.wait();

  // Check Results
  {
    std::vector<float> data = read_fp16_ndarray(runtime, out_0, {kArrLen}, {3});
    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 7);
    EXPECT_EQ(data[2], 11);
    EXPECT_EQ(data[3], 3);
    EXPECT_EQ(data[4], 7);
    EXPECT_EQ(data[5], 11);
  }
  {
    std::vector<float> data = read_fp16_ndarray(runtime, out_1, {kArrLen}, {3});
    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 17);
    EXPECT_EQ(data[2], 29);

    EXPECT_EQ(data[3], 14);
    EXPECT_EQ(data[4], 62);
    EXPECT_EQ(data[5], 110);

    EXPECT_EQ(data[6], 23);
    EXPECT_EQ(data[7], 107);
    EXPECT_EQ(data[8], 191);

    EXPECT_EQ(data[9], 32);
    EXPECT_EQ(data[10], 152);
    EXPECT_EQ(data[11], 272);
  }
}

#endif
