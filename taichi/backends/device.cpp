#include <taichi/backends/device.h>

#if TI_WITH_VULKAN
#include <taichi/backends/vulkan/vulkan_device.h>
#include <taichi/backends/interop/vulkan_cpu_interop.h>
#if TI_WITH_LLVM
#include <taichi/backends/cpu/cpu_device.h>
#endif
#if TI_WITH_CUDA
#include <taichi/backends/cuda/cuda_device.h>
#include <taichi/backends/interop/vulkan_cuda_interop.h>
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN

namespace taichi {
namespace lang {

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

#if TI_WITH_VULKAN
#if TI_WITH_LLVM
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    // TODO: support direct copy if dst itself supports host write.
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
  // Inter-device copy
#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
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

void Device::print_all_cap() const {
  const std::unordered_map<DeviceCapability, std::string> names{
      {DeviceCapability::vk_api_version, "vk_api_version"},
      {DeviceCapability::vk_has_physical_features2,
       "vk_has_physical_features2"},
      {DeviceCapability::vk_has_external_memory, "vk_has_external_memory"},
      {DeviceCapability::vk_has_surface, "vk_has_surface"},
      {DeviceCapability::vk_has_presentation, "vk_has_presentation"},
      {DeviceCapability::spirv_version, "spirv_version"},
      {DeviceCapability::spirv_has_int8, "spirv_has_int8"},
      {DeviceCapability::spirv_has_int16, "spirv_has_int16"},
      {DeviceCapability::spirv_has_int64, "spirv_has_int64"},
      {DeviceCapability::spirv_has_float16, "spirv_has_float16"},
      {DeviceCapability::spirv_has_float64, "spirv_has_float64"},
      {DeviceCapability::spirv_has_atomic_i64, "spirv_has_atomic_i64"},
      {DeviceCapability::spirv_has_atomic_float16, "spirv_has_atomic_float16"},
      {DeviceCapability::spirv_has_atomic_float16_add,
       "spirv_has_atomic_float16_add"},
      {DeviceCapability::spirv_has_atomic_float16_minmax,
       "spirv_has_atomic_float16_minmax"},
      {DeviceCapability::spirv_has_atomic_float, "spirv_has_atomic_float"},
      {DeviceCapability::spirv_has_atomic_float_add,
       "spirv_has_atomic_float_add"},
      {DeviceCapability::spirv_has_atomic_float_minmax,
       "spirv_has_atomic_float_minmax"},
      {DeviceCapability::spirv_has_atomic_float64, "spirv_has_atomic_float64"},
      {DeviceCapability::spirv_has_atomic_float64_add,
       "spirv_has_atomic_float64_add"},
      {DeviceCapability::spirv_has_atomic_float64_minmax,
       "spirv_has_atomic_float64_minmax"},
      {DeviceCapability::spirv_has_variable_ptr, "spirv_has_variable_ptr"},
      {DeviceCapability::wide_lines, "wide_lines"},
  };
  for (auto &pair : caps_) {
    TI_TRACE("DeviceCapability::{} ({}) = {}", names.at(pair.first),
             int(pair.first), pair.second);
  }
}

uint64_t *Device::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size);
  return taichi_union_cast_with_different_sizes<uint64_t *>(fetch_result_uint64(
      taichi_result_buffer_runtime_query_id, params.result_buffer));
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
