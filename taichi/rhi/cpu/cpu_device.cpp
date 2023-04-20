#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/impl_support.h"
#include "taichi/rhi/common/host_memory_pool.h"

#include "taichi/jit/jit_module.h"

namespace taichi::lang {

namespace cpu {

CpuDevice::AllocInfo CpuDevice::get_alloc_info(const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

CpuDevice::CpuDevice() {
}

RhiResult CpuDevice::allocate_memory(const AllocParams &params,
                                     DeviceAllocation *out_devalloc) {
  AllocInfo info;
  info.size = params.size;
  info.use_cached = false;

  if (info.size == 0) {
    info.ptr = nullptr;
  } else {
    info.ptr = HostMemoryPool::get_instance().allocate(
        params.size, HostMemoryPool::page_size, true /*exclusive*/);

    if (info.ptr == nullptr) {
      return RhiResult::out_of_memory;
    }
  }
  *out_devalloc = DeviceAllocation{};
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);

  return RhiResult::success;
}

DeviceAllocation CpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  DeviceAllocation alloc;
  RhiResult res = allocate_memory(params, &alloc);
  RHI_ASSERT(res == RhiResult::success &&
             "Failed to allocate memory for runtime");
  return alloc;
}

uint64_t *CpuDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);
  return reinterpret_cast<uint64_t *>(params.result_buffer[0]);
}

void CpuDevice::dealloc_memory(DeviceAllocation handle) {
  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.size == 0) {
    return;
  }
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  if (!info.use_cached) {
    HostMemoryPool::get_instance().release(info.size, info.ptr);
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

}  // namespace cpu
}  // namespace taichi::lang
