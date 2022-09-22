#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#define TI_WITH_VULKAN 1
#include "taichi/taichi_vulkan.h"

static void texture_interop_test(TiArch arch) {
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;

  ti::Runtime runtime(arch);

  ti::Texture tex_0 =
      runtime.allocate_texture2d(128, 128, TI_FORMAT_R32F, TI_NULL_HANDLE);

  TiVulkanImageInteropInfo viii{};
  ti_export_vulkan_image(runtime, tex_0.image(), &viii);
  ti_import_vulkan_image(runtime, &viii, VK_IMAGE_VIEW_TYPE_2D,
                         VK_IMAGE_LAYOUT_UNDEFINED);

  ti_track_image_ext(runtime, tex_0.image(), TI_IMAGE_LAYOUT_SHADER_READ_WRITE);
  runtime.wait();

  EXPECT_GE(ti_get_last_error(0, nullptr), TI_ERROR_SUCCESS);
}

TEST(CapiAotTest, VulkanTextureInterop) {
  if (capi::utils::is_vulkan_available()) {
    TiArch arch = TiArch::TI_ARCH_VULKAN;
    texture_interop_test(arch);
  }
}
