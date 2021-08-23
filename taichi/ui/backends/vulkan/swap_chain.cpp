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

 
void SwapChain::create_depth_resources() {
  auto size = surface_->get_size();

  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::depth32f;
  params.initial_layout = ImageLayout::undefined;
  params.x = size.first;
  params.y = size.second;
  params.z = 1;
  params.export_sharing = false;

  depth_allocation_ = app_context_->vulkan_device().create_image(params);
}

void SwapChain::resize(uint32_t width, uint32_t height){
  surface().resize(width,height);
  app_context_->vulkan_device().destroy_image(depth_allocation_);
  create_depth_resources();
}

void SwapChain::cleanup(){
  app_context_->vulkan_device().destroy_image(depth_allocation_);
  surface_.reset();
}


DeviceAllocation  SwapChain::depth_allocation(){
  return depth_allocation_;
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
