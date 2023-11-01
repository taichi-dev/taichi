#ifdef TI_WITH_VULKAN
#undef TI_WITH_VULKAN
#define TI_WITH_VULKAN 1
#endif  // TI_WITH_VULKAN

#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/taichi_cpu.h"
#include "taichi/taichi_cuda.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

#ifdef TI_WITH_LLVM
TEST_F(CapiTest, AotTestCpuBufferInterop) {
  TiArch arch = TiArch::TI_ARCH_X64;
  ti::Runtime runtime(arch);
  uint32_t size0 = 4;
  uint32_t size1 = 8;
  uint32_t vec_size = 3;

  size_t total_size = size0 * size1 * vec_size;

  const std::vector<uint32_t> shape_2d = {size0, size1};
  const std::vector<uint32_t> vec3_shape = {vec_size};

  auto ndarray = runtime.allocate_ndarray<float>(shape_2d, vec3_shape);
  std::vector<float> data(total_size, 5.0);
  ndarray.write(data);

  TiCpuMemoryInteropInfo interop_info;
  ti_export_cpu_memory(runtime, ndarray.memory().memory(), &interop_info);

  for (int i = 0; i < total_size; i++) {
    EXPECT_EQ(((float *)interop_info.ptr)[i], 5.0);
  }
}

TEST_F(CapiTest, AotTestCudaBufferInterop) {
  if (ti::is_arch_available(TI_ARCH_CUDA)) {
    TiArch arch = TiArch::TI_ARCH_CUDA;
    ti::Runtime runtime(arch);
    uint32_t size0 = 4;
    uint32_t size1 = 8;
    uint32_t vec_size = 3;

    size_t total_size = size0 * size1 * vec_size;

    const std::vector<uint32_t> shape_2d = {size0, size1};
    const std::vector<uint32_t> vec3_shape = {vec_size};

    auto ndarray = runtime.allocate_ndarray<float>(shape_2d, vec3_shape);
    std::vector<float> data(total_size, 5.0);
    ndarray.write(data);

    TiCudaMemoryInteropInfo interop_info;
    ti_export_cuda_memory(runtime, ndarray.memory().memory(), &interop_info);

    for (int i = 0; i < total_size; i++) {
      capi::utils::check_cuda_value((float *)interop_info.ptr + i, 5.0);
    }
  }
}

TEST_F(CapiTest, TestCPUImport) {
  TiArch arch = TiArch::TI_ARCH_X64;
  ti::Runtime runtime(arch);

  float data_x[4] = {1.0, 2.0, 3.0, 4.0};

  auto memory = ti_import_cpu_memory(runtime, &data_x[0], sizeof(float) * 4);

  int dim_count = 1;
  int element_count = 4;
  auto elem_type = TI_DATA_TYPE_F32;

  // prepare tiNdArray
  TiNdArray tiNdArray;
  tiNdArray.memory = memory;
  tiNdArray.shape.dim_count = dim_count;
  tiNdArray.shape.dims[0] = element_count;
  tiNdArray.elem_shape.dim_count = 0;
  tiNdArray.elem_type = elem_type;

  auto ti_memory = ti::Memory(runtime, memory, sizeof(float) * 4, false);
  // prepare ndarray
  auto ndarray = ti::NdArray<float>(std::move(ti_memory), tiNdArray);

  std::vector<float> data_out(4);
  ndarray.read(data_out);

  EXPECT_EQ(data_out[0], 1.0);
  EXPECT_EQ(data_out[1], 2.0);
  EXPECT_EQ(data_out[2], 3.0);
  EXPECT_EQ(data_out[3], 4.0);
}
#endif  // TI_WITH_LLVM

#if TI_WITH_VULKAN

static void texture_interop_test(TiArch arch) {
  ti::Runtime runtime(arch);

  ti::Texture tex_0 =
      runtime.allocate_texture2d(128, 128, TI_FORMAT_RGBA8, TI_NULL_HANDLE);

  TiVulkanImageInteropInfo viii{};
  ti_export_vulkan_image(runtime, tex_0.image(), &viii);
  ti_import_vulkan_image(runtime, &viii, VK_IMAGE_VIEW_TYPE_2D,
                         VK_IMAGE_LAYOUT_UNDEFINED);

  ti_track_image_ext(runtime, tex_0.image(), TI_IMAGE_LAYOUT_SHADER_READ_WRITE);
  runtime.wait();
}

TEST_F(CapiTest, AotTestVulkanTextureInterop) {
  if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    texture_interop_test(arch);
  }
}
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_CUDA
TEST_F(CapiTest, TestCUDAImport) {
  TiArch arch = TiArch::TI_ARCH_CUDA;
  ti::Runtime runtime(arch);

  float data_x[4] = {1.0, 2.0, 3.0, 4.0};

  void *device_array;
  size_t device_array_size = sizeof(data_x);
  capi::utils::cuda_malloc(&device_array, device_array_size);
  capi::utils::cuda_memcpy_host_to_device(device_array, data_x,
                                          device_array_size);

  auto memory = ti_import_cuda_memory(runtime, device_array, device_array_size);

  int dim_count = 1;
  int element_count = 4;
  auto elem_type = TI_DATA_TYPE_F32;

  // prepare tiNdArray
  TiNdArray tiNdArray;
  tiNdArray.memory = memory;
  tiNdArray.shape.dim_count = dim_count;
  tiNdArray.shape.dims[0] = element_count;
  tiNdArray.elem_shape.dim_count = 0;
  tiNdArray.elem_type = elem_type;

  auto ti_memory = ti::Memory(runtime, memory, sizeof(float) * 4, false);
  // prepare ndarray
  auto ndarray = ti::NdArray<float>(std::move(ti_memory), tiNdArray);

  std::vector<float> data_out(4);
  ndarray.read(data_out);

  EXPECT_EQ(data_out[0], 1.0);
  EXPECT_EQ(data_out[1], 2.0);
  EXPECT_EQ(data_out[2], 3.0);
  EXPECT_EQ(data_out[3], 4.0);
}
#endif  // TI_WITH_CUDA
