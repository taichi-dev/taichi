#include "taichi/backends/cuda/cuda_device.h"

TLANG_NAMESPACE_BEGIN


namespace cuda{


CudaDevice::AllocInfo  CudaDevice::get_alloc_info(DeviceAllocation handle){
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

DeviceAllocation CudaDevice::allocate_memory(const AllocParams &params) {
  AllocInfo info;
  //TODO: unified memory for host read/write? Need to query CUDA capabilities.
  CUDADriver::get_instance().malloc(&info.ptr, params.size);
  info.size = params.size;
  
  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

void CudaDevice::dealloc_memory(DeviceAllocation handle) {
  validate_device_alloc(handle);
  AllocInfo& info = allocations_[handle.alloc_id];
  if(info.ptr == nullptr){
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  CUDADriver::get_instance().mem_free(info.ptr);
  info.ptr = nullptr;
}

}
TLANG_NAMESPACE_END