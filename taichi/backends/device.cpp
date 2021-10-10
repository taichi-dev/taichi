#include <taichi/backends/device.h>
#include <taichi/backends/cuda/cuda_device.h>
#include <taichi/backends/cpu/cpu_device.h>
#include <taichi/backends/vulkan/vulkan_device.h>
#include <taichi/backends/interop/vulkan_cuda_interop.h>
#include <taichi/backends/interop/vulkan_cpu_interop.h>

namespace taichi {
namespace lang {

using namespace taichi::lang::vulkan;
using namespace taichi::lang::cuda;
using namespace taichi::lang::cpu;

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DevicePtr DeviceAllocation::get_ptr(uint64_t offset) const {
  return DevicePtr{{device, alloc_id}, offset};
}

Device::MemcpyCapability Device::check_memcpy_capability(DevicePtr dst,
                                                         DevicePtr src,
                                                         uint64_t size) {
  if (dst.device == src.device) {
    return Device::MemcpyCapability::Direct;
  }

#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<VulkanDevice *>(dst.device) &&
      dynamic_cast<CudaDevice *>(src.device)) {
    return Device::MemcpyCapability::Direct;
  }
#endif

  if (dynamic_cast<VulkanDevice *>(dst.device) &&
      dynamic_cast<CpuDevice *>(src.device)) {
    // TODO: support direct copy if dst itself supports host write.
    return Device::MemcpyCapability::RequiresStagingBuffer;
  }

  return Device::MemcpyCapability::RequiresHost;
}

void Device::memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Inter-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
    return;
  }
  // Intra-device copy
#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<VulkanDevice *>(dst.device) &&
      dynamic_cast<CudaDevice *>(src.device)) {
    memcpy_cuda_to_vulkan(dst, src, size);
    return;
  }
#endif

  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_staging(DevicePtr dst,
                                DevicePtr staging,
                                DevicePtr src,
                                uint64_t size) {
  // Intra-device copy
#if TI_WITH_VULKAN
  if (dynamic_cast<VulkanDevice *>(dst.device) &&
      dynamic_cast<CpuDevice *>(src.device)) {
    memcpy_cpu_to_vulkan_via_staging(dst, staging, src, size);
    return;
  }
#endif

  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_host(DevicePtr dst,
                             void *host_buffer,
                             DevicePtr src,
                             uint64_t size) {
  TI_NOT_IMPLEMENTED;
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
