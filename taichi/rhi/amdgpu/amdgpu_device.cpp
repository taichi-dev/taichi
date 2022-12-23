#include "taichi/rhi/amdgpu/amdgpu_device.h"

namespace taichi {
namespace lang {

namespace amdgpu {

AmdgpuDevice::AllocInfo AmdgpuDevice::get_alloc_info(
    const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

DeviceAllocation AmdgpuDevice::allocate_memory(const AllocParams &params) {
  AllocInfo info;

  if (params.host_read || params.host_write) {
    AMDGPUDriver::get_instance().malloc_managed(&info.ptr, params.size,
                                                HIP_MEM_ATTACH_GLOBAL);
  } else {
    AMDGPUDriver::get_instance().malloc(&info.ptr, params.size);
  }

  info.size = params.size;
  info.is_imported = false;
  info.use_cached = false;
  info.use_preallocated = false;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

DeviceAllocation AmdgpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = taichi::iroundup(params.size, taichi_page_size);
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else if (params.use_cached) {
    if (caching_allocator_ == nullptr) {
      caching_allocator_ = std::make_unique<AmdgpuCachingAllocator>(this);
    }
    info.ptr = caching_allocator_->allocate(params);
    AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);
  } else {
    info.ptr = allocate_llvm_runtime_memory_jit(params);
  }
  info.is_imported = false;
  info.use_cached = params.use_cached;
  info.use_preallocated = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

void AmdgpuDevice::dealloc_memory(DeviceAllocation handle) {
  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  TI_ASSERT(!info.is_imported);
  if (info.use_cached) {
    if (caching_allocator_ == nullptr) {
      TI_ERROR("the AmdgpuCachingAllocator is not initialized");
    }
    caching_allocator_->release(info.size, (uint64_t *)info.ptr);
  } else if (!info.use_preallocated) {
    AMDGPUDriver::get_instance().mem_free(info.ptr);
    info.ptr = nullptr;
  }
}

RhiResult AmdgpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];
  // FIXME: there should be a better way to do this...
  AMDGPUDriver::get_instance().memcpy_device_to_host(info.mapped, info.ptr, size);
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

uint64 AmdgpuDevice::fetch_result_uint64(int i, uint64 *result_buffer) {
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  uint64 ret;
  AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                     sizeof(uint64));
  return ret;
}
}  // namespace amdgpu
}  // namespace lang
}  // namespace taichi
