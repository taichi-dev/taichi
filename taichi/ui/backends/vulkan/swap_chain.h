#pragma once

#include <taichi/rhi/device.h>

TI_UI_NAMESPACE_BEGIN
namespace vulkan {

class TI_DLL_EXPORT SwapChain {
 public:
  void init(class AppContext *app_context);
  uint32_t width();
  uint32_t height();

  taichi::lang::Surface &surface();
  taichi::lang::DeviceAllocation depth_allocation();

  void resize(uint32_t width, uint32_t height);

  bool copy_depth_buffer_to_ndarray(taichi::lang::DevicePtr &arr_dev_ptr);

  std::vector<float> &dump_depth_buffer();

  std::vector<uint32_t> &dump_image_buffer();

  void write_image(const std::string &filename);

  void cleanup();

 private:
  void create_depth_resources();

  taichi::lang::DeviceAllocation depth_allocation_;

  std::unique_ptr<taichi::lang::Surface> surface_;

  std::vector<float> depth_buffer_data_;

  std::vector<uint32_t> image_buffer_data_;

  class AppContext *app_context_;

  uint32_t curr_width_;
  uint32_t curr_height_;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
