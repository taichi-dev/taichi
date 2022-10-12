#ifdef TI_WITH_VULKAN
#undef TI_WITH_VULKAN
#define TI_WITH_VULKAN 1
#endif  // TI_WITH_VULKAN

#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

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

  EXPECT_GE(ti_get_last_error(0, nullptr), TI_ERROR_SUCCESS);
}

TEST_F(CapiTest, AotTestVulkanTextureInterop) {
  if (capi::utils::is_vulkan_available()) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    texture_interop_test(arch);
  }
}

#endif  // TI_WITH_VULKAN
