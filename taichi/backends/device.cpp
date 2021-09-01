#include <taichi/backends/device.h>

namespace taichi {
namespace lang {

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DevicePtr DeviceAllocation::get_ptr(uint64_t offset) const {
  return DevicePtr{device, alloc_id, offset};
}

void Device::memcpy(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Inter-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
  }
  // Intra-device copy
#if TI_WITH_VULKAN && TI_WITH_CUDA

#endif
}

void GraphicsDevice::image_transition(DeviceAllocation img,
                                      ImageLayout old_layout,
                                      ImageLayout new_layout) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->image_transition(img, old_layout, new_layout);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::buffer_to_image(DeviceAllocation dst_img,
                                     DevicePtr src_buf,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->buffer_to_image(dst_img, src_buf, img_layout, params);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::image_to_buffer(DevicePtr dst_buf,
                                     DeviceAllocation src_img,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->image_to_buffer(dst_buf, src_img, img_layout, params);
  stream->submit_synced(cmd_list.get());
}

}  // namespace lang
}  // namespace taichi
