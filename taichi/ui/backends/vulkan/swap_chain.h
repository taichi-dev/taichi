#pragma once

#include <taichi/backends/device.h>

TI_UI_NAMESPACE_BEGIN
namespace vulkan {

class SwapChain {
 public:
  void init(class AppContext *app_context);
  uint32_t width();
  uint32_t height();
  VkFormat depth_format();
  taichi::lang::Surface& surface();
  VkFramebuffer framebuffer(VkRenderPass render_pass);
  taichi::lang::DeviceAllocation depth_allocation();

 private:

  void create_depth_resources();
 
  std::vector<VkFramebuffer> swap_chain_framebuffers_;

  // TODO: make the device api support allocating images.
  VkImage depth_image_;
  VkDeviceMemory depth_image_memory_;
  VkImageView depth_image_view_;
  VkFormat depth_format_;
  taichi::lang::DeviceAllocation depth_allocation_;

  std::unique_ptr<taichi::lang::Surface> surface_; 


  class AppContext *app_context_;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
