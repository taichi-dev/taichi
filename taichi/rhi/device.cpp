#include <taichi/rhi/device.h>

#if TI_WITH_VULKAN
#include <taichi/rhi/vulkan/vulkan_device.h>
#include <taichi/rhi/interop/vulkan_cpu_interop.h>
#if TI_WITH_LLVM
#include <taichi/rhi/cpu/cpu_device.h>
#endif
#if TI_WITH_CUDA
#include <taichi/rhi/cuda/cuda_device.h>
#include <taichi/rhi/interop/vulkan_cuda_interop.h>
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN

namespace taichi::lang {

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DeviceImageGuard::~DeviceImageGuard() {
  dynamic_cast<GraphicsDevice *>(device)->destroy_image(*this);
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

#if TI_WITH_VULKAN
#if TI_WITH_LLVM
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    // TODO: support direct copy if dst itself supports host write.
    return Device::MemcpyCapability::RequiresStagingBuffer;
  } else if (dynamic_cast<cpu::CpuDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::RequiresStagingBuffer;
  }
#endif
#if TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    // FIXME: direct copy isn't always possible.
    // The vulkan buffer needs export_sharing turned on.
    // Otherwise, needs staging buffer
    return Device::MemcpyCapability::Direct;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::Direct;
  }
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN
  return Device::MemcpyCapability::RequiresHost;
}

void Device::memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Intra-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
    return;
  }
#if TI_WITH_VULKAN && TI_WITH_LLVM
  // cross-device copy directly
  else if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
           dynamic_cast<cpu::CpuDevice *>(src.device)) {
    memcpy_cpu_to_vulkan(dst, src, size);
    return;
  }
#endif
#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    memcpy_cuda_to_vulkan(dst, src, size);
    return;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    memcpy_vulkan_to_cuda(dst, src, size);
    return;
  }
#endif
  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_staging(DevicePtr dst,
                                DevicePtr staging,
                                DevicePtr src,
                                uint64_t size) {
  // Inter-device copy
#if defined(TI_WITH_VULKAN) && defined(TI_WITH_LLVM)
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
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
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_transition(img, old_layout, new_layout);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::buffer_to_image(DeviceAllocation dst_img,
                                     DevicePtr src_buf,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->buffer_to_image(dst_img, src_buf, img_layout, params);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::image_to_buffer(DevicePtr dst_buf,
                                     DeviceAllocation src_img,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_to_buffer(dst_buf, src_img, img_layout, params);
  stream->submit_synced(cmd_list.get());
}

}  // namespace taichi::lang
