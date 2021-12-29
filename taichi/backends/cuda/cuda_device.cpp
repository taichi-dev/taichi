#include "taichi/backends/cuda/cuda_device.h"

namespace taichi {
namespace lang {

namespace cuda {

CudaDevice::AllocInfo CudaDevice::get_alloc_info(DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

DeviceAllocation CudaDevice::allocate_memory(const AllocParams &params) {
  AllocInfo info;

  if (params.host_read || params.host_write) {
    CUDADriver::get_instance().malloc_managed(&info.ptr, params.size,
                                              CU_MEM_ATTACH_GLOBAL);
  } else {
    CUDADriver::get_instance().malloc(&info.ptr, params.size);
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

DeviceAllocation CudaDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = taichi::iroundup(params.size, taichi_page_size);
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else if (params.use_cached) {
    if (caching_allocator_ == nullptr) {
      caching_allocator_ = std::make_unique<CudaCachingAllocator>(this);
    }
    info.ptr = caching_allocator_->allocate(params);
    CUDADriver::get_instance().memset((void *)info.ptr, 0, info.size);
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

void CudaDevice::dealloc_memory(DeviceAllocation handle) {
  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  TI_ASSERT(!info.is_imported);
  if (info.use_cached) {
    if (caching_allocator_ == nullptr) {
      TI_ERROR("the CudaCachingAllocator is not initialized");
    }
    caching_allocator_->release(info.size, (uint64_t *)info.ptr);
  } else if (!info.use_preallocated) {
    CUDADriver::get_instance().mem_free(info.ptr);
    info.ptr = nullptr;
  }
}

DeviceAllocation CudaDevice::import_memory(void *ptr, size_t size) {
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

uint64 CudaDevice::fetch_result_uint64(int i, uint64 *result_buffer) {
  CUDADriver::get_instance().stream_synchronize(nullptr);
  uint64 ret;
  CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                   sizeof(uint64));
  return ret;
}

CudaCommandList::CudaCommandList(CudaDevice *ti_device)
  : ti_device_(ti_device){}

void CudaCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  auto ptr_d = ti_device_->get_alloc_info(ptr).ptr;
  if (ptr_d == nullptr) {
    TI_ERROR("the DevicePtr is null");
  }
  CUDADriver::get_instance().memsetd32((void *)ptr_d, data, size);
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
