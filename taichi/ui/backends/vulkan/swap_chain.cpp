#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/util/image_io.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

void SwapChain::init(class AppContext *app_context) {
  app_context_ = app_context;
  SurfaceConfig config;
  config.vsync = app_context_->config.vsync;
  config.window_handle = app_context_->taichi_window();
  config.width = app_context_->config.width;
  config.height = app_context_->config.height;
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

bool SwapChain::copy_depth_buffer_to_ndarray(
    taichi::lang::DevicePtr &arr_dev_ptr) {
  auto [w, h] = surface_->get_size();
  size_t copy_size = w * h * 4;

  Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
      arr_dev_ptr, depth_allocation_.get_ptr(), copy_size);

  auto &device = app_context_->device();
  auto *stream = device.get_graphics_stream();
  std::unique_ptr<CommandList> cmd_list{nullptr};

  if (memcpy_cap == Device::MemcpyCapability::Direct) {
    Device::AllocParams params{copy_size, /*host_wrtie*/ false,
                               /*host_read*/ false, /*export_sharing*/ true,
                               AllocUsage::Uniform};

    auto depth_staging_buffer = device.allocate_memory(params);

    BufferImageCopyParams copy_params;
    copy_params.image_extent.x = w;
    copy_params.image_extent.y = h;
    copy_params.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
    cmd_list = stream->new_command_list();
    cmd_list->image_transition(depth_allocation_, ImageLayout::depth_attachment,
                               ImageLayout::transfer_src);
    cmd_list->image_to_buffer(depth_staging_buffer.get_ptr(), depth_allocation_,
                              ImageLayout::transfer_src, copy_params);
    cmd_list->image_transition(depth_allocation_, ImageLayout::transfer_src,
                               ImageLayout::depth_attachment);
    stream->submit_synced(cmd_list.get());
    Device::memcpy_direct(arr_dev_ptr, depth_staging_buffer.get_ptr(),
                          copy_size);

    device.dealloc_memory(depth_staging_buffer);

  } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
    DeviceAllocation depth_buffer = surface_->get_depth_data(depth_allocation_);
    DeviceAllocation field_buffer(arr_dev_ptr);
    float *src_ptr = (float *)app_context_->device().map(depth_buffer);
    float *dst_ptr = (float *)arr_dev_ptr.device->map(field_buffer);
    memcpy(dst_ptr, src_ptr, copy_size);
    app_context_->device().unmap(depth_buffer);
    arr_dev_ptr.device->unmap(field_buffer);
  } else {
    TI_NOT_IMPLEMENTED;
    return 0;
  }
  return 1;
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

std::vector<uint32_t> &SwapChain::dump_image_buffer() {
  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  image_buffer_data_.clear();
  image_buffer_data_.resize(w * h);
  DeviceAllocation img_buffer = surface_->get_image_data();
  unsigned char *ptr = (unsigned char *)app_context_->device().map(img_buffer);
  auto format = surface_->image_format();
  uint32_t *u32ptr = (uint32_t *)ptr;
  if (format == BufferFormat::bgra8 || format == BufferFormat::bgra8srgb) {
    TI_TRACE(
        "Converting BGRA8 to RGBA8 for converting image format to a standard "
        "format");
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        auto pixel = u32ptr[j * w + i];
        image_buffer_data_[j * w + i] =
            ((pixel << 16) & 0xFF0000) | (pixel & 0x0000FF00) |
            ((pixel >> 16) & 0xFF) | (pixel & 0xFF000000);
      }
    }
  } else {
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        image_buffer_data_[j * w + i] = u32ptr[j * w + i];
      }
    }
  }
  app_context_->device().unmap(img_buffer);
  return image_buffer_data_;
}

void SwapChain::write_image(const std::string &filename) {
  dump_image_buffer();
  imwrite(filename, (size_t)image_buffer_data_.data(), curr_width_,
          curr_height_, 4);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
