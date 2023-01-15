#include "taichi/rhi/interop/vulkan_cpu_interop.h"
#include "taichi/rhi/cpu/cpu_device.h"

#if TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif  // TI_WITH_VULKAN

#include <unordered_map>

namespace taichi::lang {

#if TI_WITH_VULKAN && defined(TI_WITH_LLVM)

using namespace taichi::lang::vulkan;
using namespace taichi::lang::cpu;

void memcpy_cpu_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Note that `dst` must point to host-visible memory, if `dst` point to
  // device-local memory, please choose to use `memcpy_via_staging`.
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CpuDevice *cpu_dev = dynamic_cast<CpuDevice *>(src.device);

  DeviceAllocation src_alloc(src);

  CpuDevice::AllocInfo src_alloc_info = cpu_dev->get_alloc_info(src_alloc);

  void *dst_ptr{nullptr};
  TI_ASSERT(vk_dev->map_range(dst, size, &dst_ptr) == RhiResult::success);
  void *src_ptr = (uint8_t *)src_alloc_info.ptr + src.offset;

  memcpy(dst_ptr, src_ptr, size);
  vk_dev->unmap(dst);
}

void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr staging,
                                      DevicePtr src,
                                      uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CpuDevice *cpu_dev = dynamic_cast<CpuDevice *>(src.device);

  DeviceAllocation src_alloc(src);

  CpuDevice::AllocInfo src_alloc_info = cpu_dev->get_alloc_info(src_alloc);

  void *dst_ptr{nullptr};
  TI_ASSERT(vk_dev->map_range(staging, size, &dst_ptr) == RhiResult::success);
  void *src_ptr = (uint8_t *)src_alloc_info.ptr + src.offset;

  memcpy(dst_ptr, src_ptr, size);
  vk_dev->unmap(staging);

  auto stream = vk_dev->get_compute_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->buffer_copy(dst, staging, size);
  stream->submit_synced(cmd_list.get());
}

#else
void memcpy_cpu_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}
void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr stagin,
                                      DevicePtr src,
                                      uint64_t size) {
  TI_NOT_IMPLEMENTED;
}
#endif  // TI_WITH_VULKAN

}  // namespace taichi::lang
