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

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

DeviceAllocation CudaDevice::allocate_memory_runtime(const AllocParams &params,
                                                     JITModule *runtime_jit,
                                                     LLVMRuntime *runtime,
                                                     uint64 *result_buffer) {
  AllocInfo info;
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else {
    info.ptr = allocate_llvm_runtime_memory_jit(runtime_jit, runtime,
                                                params.size, result_buffer);
  }
  info.size = params.size;
  info.is_imported = false;

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
  CUDADriver::get_instance().mem_free(info.ptr);
  info.ptr = nullptr;
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
}  // namespace cuda
}  // namespace lang
}  // namespace taichi
