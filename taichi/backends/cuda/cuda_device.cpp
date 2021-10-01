#include "taichi/backends/cuda/cuda_device.h"
#include <iostream>

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

  curr_mem += info.size;
  if (curr_mem > max_mem) {
    max_mem = curr_mem;
    std::cout << "Max CUDA memory allocation: " << max_mem << std::endl;
  }

  info.size = params.size;

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
  CUDADriver::get_instance().mem_free(info.ptr);
  curr_mem -= info.size;
  info.ptr = nullptr;
}

}  // namespace cuda
}  // namespace lang

}  // namespace taichi
