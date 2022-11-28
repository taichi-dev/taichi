#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"

#ifdef ANDROID
#include <android/native_window_jni.h>
#else
#include <GLFW/glfw3.h>
#endif

namespace taichi::lang {
namespace vulkan {

class VulkanSurface : public Surface {
 public:
  VulkanSurface(VulkanDevice *device, const SurfaceConfig &config);
  ~VulkanSurface() override;

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;

  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::pair<uint32_t, uint32_t> get_size() override;
  int get_image_count() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;

  DeviceAllocation get_depth_data(DeviceAllocation &depth_alloc) override;
  DeviceAllocation get_image_data() override;

 private:
  void create_swap_chain();
  void destroy_swap_chain();

  SurfaceConfig config_;

  VulkanDevice *device_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swapchain_;
  vkapi::IVkSemaphore image_available_;
#ifdef ANDROID
  ANativeWindow *window_;
#else
  GLFWwindow *window_;
#endif
  BufferFormat image_format_;

  uint32_t image_index_{0};

  uint32_t width_{0};
  uint32_t height_{0};

  std::vector<DeviceAllocation> swapchain_images_;

  // DeviceAllocation screenshot_image_{kDeviceNullAllocation};
  DeviceAllocation depth_buffer_{kDeviceNullAllocation};
  DeviceAllocation screenshot_buffer_{kDeviceNullAllocation};
};

} // namespace vulkan
} // namespace taichi::lang
