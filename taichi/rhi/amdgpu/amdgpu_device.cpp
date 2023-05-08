#include "taichi/rhi/amdgpu/amdgpu_device.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

#include "taichi/jit/jit_module.h"

namespace taichi {
namespace lang {

namespace amdgpu {

AmdgpuDevice::AmdgpuDevice() {
  // Initialize the device memory pool
  DeviceMemoryPool::get_instance(false /*merge_upon_release*/);
}

AmdgpuDevice::AllocInfo AmdgpuDevice::get_alloc_info(
    const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

RhiResult AmdgpuDevice::allocate_memory(const AllocParams &params,
                                        DeviceAllocation *out_devalloc) {
  AllocInfo info;

  auto &mem_pool = DeviceMemoryPool::get_instance();

  bool managed = params.host_read || params.host_write;
  void *ptr =
      mem_pool.allocate(params.size, DeviceMemoryPool::page_size, managed);
  if (ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  info.ptr = ptr;
  info.size = params.size;
  info.is_imported = false;
  info.use_cached = false;
  info.use_preallocated = false;

  if (info.ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  *out_devalloc = DeviceAllocation{};
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);
  return RhiResult::success;
}

DeviceAllocation AmdgpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = taichi::iroundup(params.size, taichi_page_size);
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else {
    info.ptr =
        DeviceMemoryPool::get_instance().allocate_with_cache(this, params);
    TI_ASSERT(info.ptr != nullptr);

    AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);
  }
  info.is_imported = false;
  info.use_cached = true;
  info.use_preallocated = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

uint64_t *AmdgpuDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  uint64 *ret{nullptr};
  AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, params.result_buffer,
                                                     sizeof(uint64));
  return ret;
}

void AmdgpuDevice::dealloc_memory(DeviceAllocation handle) {
  // After reset, all allocations are invalid
  if (allocations_.empty()) {
    return;
  }

  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  TI_ASSERT(!info.is_imported);
  if (info.use_cached) {
    DeviceMemoryPool::get_instance().release(info.size, (uint64_t *)info.ptr,
                                             false);
  } else if (!info.use_preallocated) {
    DeviceMemoryPool::get_instance().release(info.size, info.ptr);
    info.ptr = nullptr;
  }
}

RhiResult AmdgpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];
  // FIXME: there should be a better way to do this...
  AMDGPUDriver::get_instance().memcpy_device_to_host(info.mapped, info.ptr,
                                                     size);
  *mapped_ptr = info.mapped;
  return RhiResult::success;
}

void AmdgpuDevice::unmap(DeviceAllocation alloc) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  AMDGPUDriver::get_instance().memcpy_host_to_device(info.ptr, info.mapped,
                                                     info.size);
  delete[] static_cast<char *>(info.mapped);
  return;
}

void AmdgpuDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  void *dst_ptr =
      static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr =
      static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;
  AMDGPUDriver::get_instance().memcpy_device_to_device(dst_ptr, src_ptr, size);
}

DeviceAllocation AmdgpuDevice::import_memory(void *ptr, size_t size) {
  AllocInfo info;
  info.ptr = ptr;
  info.size = size;
  info.is_imported = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

}  // namespace amdgpu
}  // namespace lang
}  // namespace taichi
