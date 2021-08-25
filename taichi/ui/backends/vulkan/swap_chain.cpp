#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

void SwapChain::init(class AppContext *app_context) {
  app_context_ = app_context;
  SurfaceConfig config;
  config.vsync = app_context_->config.vsync;
  config.window_handle = app_context_->glfw_window();

  surface_ = app_context_->device().create_surface(config);
  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  create_depth_resources();
}

void SwapChain::create_depth_resources() {
  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::depth32f;
  params.initial_layout = ImageLayout::undefined;
  params.x = curr_width_;
  params.y = curr_height_;
  params.export_sharing = false;

  depth_allocation_ = app_context_->device().create_image(params);
}

void SwapChain::resize(uint32_t width, uint32_t height) {
  surface().resize(width, height);
  app_context_->device().destroy_image(depth_allocation_);
  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  create_depth_resources();
}

void SwapChain::cleanup() {
  app_context_->device().destroy_image(depth_allocation_);
  surface_.reset();
}

DeviceAllocation SwapChain::depth_allocation() {
  return depth_allocation_;
}

uint32_t SwapChain::width() {
  return curr_width_;
}
uint32_t SwapChain::height() {
  return curr_height_;
}
taichi::lang::Surface &SwapChain::surface() {
  return *(surface_.get());
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
