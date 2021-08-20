#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

void SwapChain::init(class AppContext *app_context) {
  app_context_ = app_context;
  SurfaceConfig config;
  config.width = app_context_->config.width;
  config.height = app_context_->config.height;
  config.vsync = app_context_->config.vsync;
  config.window_handle = app_context_->glfw_window();
  
  surface_ = app_context_->vulkan_device().create_surface(config);

  create_depth_resources(); 
}


VkFramebuffer SwapChain::framebuffer(VkRenderPass render_pass){
  DeviceAllocation image_alloc = surface_->get_target_image();
  auto [img,view,format] = app_context_->vulkan_device().get_vk_image(image_alloc);
  VulkanFramebufferDesc desc;
  desc.width = width();
  desc.height = height();
  desc.attachments = {view,depth_image_view_};
  desc.renderpass = render_pass;
  return app_context_->vulkan_device().get_framebuffer(desc);
}

 
 VkFormat SwapChain::depth_format(){
   return depth_format_;
 }
void SwapChain::create_depth_resources() {
   depth_format_ = find_depth_format(app_context_->physical_device());
  auto size = surface_->get_size();
  create_image(
      2, size.first, size.second, 1, depth_format_,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image_, depth_image_memory_,
      app_context_->device(), app_context_->physical_device());
  depth_image_view_ =
      create_image_view(2, depth_image_, depth_format_,
                        VK_IMAGE_ASPECT_DEPTH_BIT, app_context_->device());

  depth_allocation_ = app_context_->vulkan_device().import_vk_image(depth_image_, depth_image_view_, depth_format_);
}

 
uint32_t  SwapChain::width(){
   return surface_->get_size().first;
}
uint32_t  SwapChain::height(){
  return surface_->get_size().second;
}
 taichi::lang::Surface& SwapChain::surface(){
   return *(surface_.get());
 }
 
 
 
}  // namespace vulkan

TI_UI_NAMESPACE_END
