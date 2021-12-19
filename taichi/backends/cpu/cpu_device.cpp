#include "taichi/backends/cpu/cpu_device.h"

namespace taichi {
namespace lang {

namespace cpu {

CpuDevice::AllocInfo CpuDevice::get_alloc_info(DeviceAllocation handle) {
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
  AllocInfo info;
  info.ptr = allocate_llvm_runtime_memory_jit(params);
  // TODO: Add caching allocator
  info.size = params.size;
  info.use_cached = params.use_cached;
  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
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
}  // namespace lang
}  // namespace taichi
