#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/util/image_io.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

void SwapChain::init(class AppContext *app_context) {
  app_context_ = app_context;
  SurfaceConfig config;
  config.vsync = app_context_->config.vsync;
  config.native_surface_handle = app_context_->get_native_surface();
  config.width = app_context_->config.width;
  config.height = app_context_->config.height;
  surface_ = app_context_->device().create_surface(config);
  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  create_depth_resources();
  create_image_resources();
}

void SwapChain::create_depth_resources() {
  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::depth32f;
  params.initial_layout = ImageLayout::undefined;
  params.x = curr_width_;
  params.y = curr_height_;
  params.export_sharing = false;
  params.usage = ImageAllocUsage::Attachment | ImageAllocUsage::Sampled;

  depth_allocation_ = app_context_->device().create_image_unique(params);

  auto [w, h] = surface_->get_size();
  size_t size_bytes = size_t(w * h) * sizeof(float);
  Device::AllocParams params_buff{size_bytes, /*host_wrtie*/ false,
                                  /*host_read*/ true, /*export_sharing*/ false,
                                  AllocUsage::Uniform};
  auto [buf, res] = app_context_->device().allocate_memory_unique(params_buff);
  RHI_ASSERT(res == RhiResult::success);
  depth_buffer_ = std::move(buf);
}

void SwapChain::create_image_resources() {
  auto [w, h] = surface_->get_size();
  size_t size_bytes = size_t(w * h) * sizeof(uint8_t) * 4;
  Device::AllocParams params{size_bytes, /*host_wrtie*/ false,
                             /*host_read*/ true, /*export_sharing*/ false,
                             AllocUsage::Uniform};
  auto [buf, res] = app_context_->device().allocate_memory_unique(params);
  RHI_ASSERT(res == RhiResult::success);
  screenshot_buffer_ = std::move(buf);
}

void SwapChain::resize(uint32_t width, uint32_t height) {
  surface().resize(width, height);
  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  create_depth_resources();
  create_image_resources();
}

bool SwapChain::copy_depth_buffer_to_ndarray(
    taichi::lang::DevicePtr &arr_dev_ptr) {
  auto [w, h] = surface_->get_size();
  size_t copy_size = w * h * 4;

  Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
      arr_dev_ptr, depth_allocation_->get_ptr(), copy_size);

  auto &device = app_context_->device();
  auto *stream = device.get_graphics_stream();
  std::unique_ptr<CommandList> cmd_list{nullptr};

  device.wait_idle();

  if (memcpy_cap == Device::MemcpyCapability::Direct) {
    Device::AllocParams params{copy_size, /*host_wrtie*/ false,
                               /*host_read*/ false, /*export_sharing*/ true,
                               AllocUsage::Uniform};

    auto [depth_staging_buffer, res_alloc] =
        device.allocate_memory_unique(params);
    TI_ASSERT(res_alloc == RhiResult::success);

    BufferImageCopyParams copy_params;
    copy_params.image_extent.x = w;
    copy_params.image_extent.y = h;
    copy_params.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
    auto [cmd_list, res_cmdlist] = stream->new_command_list_unique();
    assert(res_cmdlist == RhiResult::success &&
           "Failed to allocate command list");
    cmd_list->image_transition(*depth_allocation_,
                               ImageLayout::depth_attachment,
                               ImageLayout::transfer_src);
    cmd_list->image_to_buffer(depth_staging_buffer->get_ptr(),
                              *depth_allocation_, ImageLayout::transfer_src,
                              copy_params);
    cmd_list->image_transition(*depth_allocation_, ImageLayout::transfer_src,
                               ImageLayout::depth_attachment);
    stream->submit_synced(cmd_list.get());
    Device::memcpy_direct(arr_dev_ptr, depth_staging_buffer->get_ptr(),
                          copy_size);
  } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
    BufferImageCopyParams copy_params;
    copy_params.image_extent.x = w;
    copy_params.image_extent.y = h;
    copy_params.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
    auto [cmd_list, res] = stream->new_command_list_unique();
    assert(res == RhiResult::success && "Failed to allocate command list");
    cmd_list->image_transition(*depth_allocation_,
                               ImageLayout::depth_attachment,
                               ImageLayout::transfer_src);
    cmd_list->image_to_buffer(depth_buffer_->get_ptr(), *depth_allocation_,
                              ImageLayout::transfer_src, copy_params);
    cmd_list->image_transition(*depth_allocation_, ImageLayout::transfer_src,
                               ImageLayout::depth_attachment);
    stream->submit_synced(cmd_list.get());
    DeviceAllocation field_buffer(arr_dev_ptr);
    void *src_ptr{nullptr}, *dst_ptr{nullptr};
    TI_ASSERT(app_context_->device().map(*depth_buffer_, &src_ptr) ==
              RhiResult::success);
    TI_ASSERT(arr_dev_ptr.device->map(field_buffer, &dst_ptr) ==
              RhiResult::success);
    memcpy(dst_ptr, src_ptr, copy_size);
    app_context_->device().unmap(*depth_buffer_);
    arr_dev_ptr.device->unmap(field_buffer);
  } else {
    TI_NOT_IMPLEMENTED;
    return false;
  }
  return true;
}

SwapChain::~SwapChain() {
}

DeviceAllocation SwapChain::depth_allocation() {
  return *depth_allocation_;
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
  auto &device = app_context_->device();

  device.wait_idle();

  TI_INFO("Dumping image buffer...");

  auto *stream = device.get_graphics_stream();

  auto [w, h] = surface_->get_size();
  curr_width_ = w;
  curr_height_ = h;
  image_buffer_data_.clear();
  image_buffer_data_.resize(w * h);
  DeviceAllocation img_alloc = surface_->get_target_image();

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = w;
  copy_params.image_extent.y = h;
  copy_params.image_aspect_flag = VK_IMAGE_ASPECT_COLOR_BIT;
  auto [cmd_list, res] = stream->new_command_list_unique();
  assert(res == RhiResult::success && "Failed to allocate command list");
  cmd_list->image_transition(img_alloc, ImageLayout::present_src,
                             ImageLayout::transfer_src);
  // TODO: directly map the image to cpu memory
  cmd_list->image_to_buffer(screenshot_buffer_->get_ptr(), img_alloc,
                            ImageLayout::transfer_src, copy_params);
  cmd_list->image_transition(img_alloc, ImageLayout::transfer_src,
                             ImageLayout::present_src);
  stream->submit_synced(cmd_list.get());

  void *ptr{nullptr};
  TI_ASSERT(app_context_->device().map(*screenshot_buffer_, &ptr) ==
            RhiResult::success);
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
  app_context_->device().unmap(*screenshot_buffer_);
  return image_buffer_data_;
}

void SwapChain::write_image(const std::string &filename) {
  dump_image_buffer();
  imwrite(filename, (size_t)image_buffer_data_.data(), curr_width_,
          curr_height_, 4);
}

}  // namespace vulkan

}  // namespace taichi::ui
