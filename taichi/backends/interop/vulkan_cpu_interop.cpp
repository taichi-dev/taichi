#include "taichi/backends/interop/vulkan_cpu_interop.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/backends/cpu/cpu_device.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/jit/jit_session.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/file_sequence_writer.h"

#include <unordered_map>

namespace taichi {
namespace lang {

using namespace taichi::lang::vulkan;
using namespace taichi::lang::cpu;

#if TI_WITH_VULKAN

void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr staging,
                                      DevicePtr src,
                                      uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CpuDevice *cpu_dev = dynamic_cast<CpuDevice *>(src.device);

  DeviceAllocation dst_alloc(dst);
  DeviceAllocation src_alloc(src);

  CpuDevice::AllocInfo src_alloc_info = cpu_dev->get_alloc_info(src_alloc);

  unsigned char *dst_ptr = (unsigned char *)(vk_dev->map_range(staging, size));
  unsigned char *src_ptr = (unsigned char *)src_alloc_info.ptr + src.offset;

  memcpy(dst_ptr, src_ptr, size);
  vk_dev->unmap(staging);

  auto stream = vk_dev->get_compute_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->buffer_copy(dst, staging, size);
  stream->submit_synced(cmd_list.get());
}

#else
void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr stagin,
                                      DevicePtr src,
                                      uint64_t size) {
  TI_NOT_IMPLEMENTED;
}
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

}  // namespace lang
}  // namespace taichi
