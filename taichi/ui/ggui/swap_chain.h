#pragma once

#include <taichi/rhi/device.h>

namespace taichi::ui {
namespace vulkan {

class TI_DLL_EXPORT SwapChain {
 public:
  void init(class AppContext *app_context);
  ~SwapChain();

  uint32_t width();
  uint32_t height();

  taichi::lang::Surface &surface();
  taichi::lang::DeviceAllocation depth_allocation();

  void resize(uint32_t width, uint32_t height);

  bool copy_depth_buffer_to_ndarray(taichi::lang::DevicePtr &arr_dev_ptr);

  std::vector<uint32_t> &dump_image_buffer();

  void write_image(const std::string &filename);

 private:
  void create_depth_resources();
  void create_image_resources();

  std::unique_ptr<taichi::lang::Surface> surface_{nullptr};
  taichi::lang::DeviceImageUnique depth_allocation_{nullptr};

  std::vector<uint32_t> image_buffer_data_;

  taichi::lang::DeviceAllocationUnique depth_buffer_{nullptr};
  taichi::lang::DeviceAllocationUnique screenshot_buffer_{nullptr};

  class AppContext *app_context_;

  uint32_t curr_width_;
  uint32_t curr_height_;
};

}  // namespace vulkan

}  // namespace taichi::ui
