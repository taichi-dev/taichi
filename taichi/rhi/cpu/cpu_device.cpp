#include "taichi/rhi/cpu/cpu_device.h"

namespace taichi::lang {

namespace cpu {

CpuDevice::AllocInfo CpuDevice::get_alloc_info(const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

DeviceAllocation CpuDevice::allocate_memory(const AllocParams &params) {
  AllocInfo info;

  auto vm = std::make_unique<VirtualMemoryAllocator>(params.size);
  info.ptr = vm->ptr;
  info.size = vm->size;
  info.use_cached = false;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  virtual_memories_[alloc.alloc_id] = std::move(vm);
  return alloc;
}

DeviceAllocation CpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  return allocate_memory(params);
}

void CpuDevice::dealloc_memory(DeviceAllocation handle) {
  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  if (!info.use_cached) {
    // Use at() to ensure that the memory is allocated, and not imported
    virtual_memories_.at(handle.alloc_id).reset();
    info.ptr = nullptr;
  }
}

RhiResult CpuDevice::upload_data(DevicePtr *device_ptr,
                                 const void **data,
                                 size_t *size,
                                 int num_alloc) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    AllocInfo &info = allocations_[device_ptr[i].alloc_id];
    memcpy((uint8_t *)info.ptr + device_ptr[i].offset, data[i], size[i]);
  }

  return RhiResult::success;
}

RhiResult CpuDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    AllocInfo &info = allocations_[device_ptr[i].alloc_id];
    memcpy(data[i], (uint8_t *)info.ptr + device_ptr[i].offset, size[i]);
  }

  return RhiResult::success;
}

RhiResult CpuDevice::map_range(DevicePtr ptr,
                               uint64_t size,
                               void **mapped_ptr) {
  AllocInfo &info = allocations_[ptr.alloc_id];
  if (info.ptr == nullptr) {
    return RhiResult::error;
  }
  *mapped_ptr = (uint8_t *)info.ptr + ptr.offset;
  return RhiResult::success;
}

RhiResult CpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  if (info.ptr == nullptr) {
    return RhiResult::error;
  }
  *mapped_ptr = info.ptr;
  return RhiResult::success;
}

void CpuDevice::unmap(DeviceAllocation alloc) {
  return;
}

void CpuDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  void *dst_ptr =
      static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr =
      static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;
  std::memcpy(dst_ptr, src_ptr, size);
}

DeviceAllocation CpuDevice::import_memory(void *ptr, size_t size) {
  AllocInfo info;
  info.ptr = ptr;
  info.size = size;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

uint64 CpuDevice::fetch_result_uint64(int i, uint64 *result_buffer) {
  uint64 ret = result_buffer[i];
  return ret;
}

}  // namespace cpu
}  // namespace taichi::lang
