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
#endif

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
