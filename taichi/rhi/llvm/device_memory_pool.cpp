#include "taichi/rhi/llvm/device_memory_pool.h"

#include <memory>

#ifdef TI_WITH_AMDGPU
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#endif

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

#if defined(TI_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "taichi/platform/windows/windows.h"
#endif

namespace taichi::lang {

DeviceMemoryPool::DeviceMemoryPool() {
}

void *DeviceMemoryPool::allocate(std::size_t size,
                                 std::size_t alignment,
                                 bool managed) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  return allocate_raw_memory(size, managed);
}

void DeviceMemoryPool::release(std::size_t size, void *ptr, bool release_raw) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  deallocate_raw_memory(ptr);
}

void *DeviceMemoryPool::allocate_raw_memory(std::size_t size, bool managed) {
  /*
    Be aware that this methods is not protected by the mutex.

    allocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  void *ptr = nullptr;

#if TI_WITH_CUDA
  if (!managed) {
    CUDADriver::get_instance().malloc(&ptr, size);
  } else {
    CUDADriver::get_instance().malloc_managed(&ptr, size, CU_MEM_ATTACH_GLOBAL);
  }
#elif TI_WITH_AMDGPU
  if (!managed) {
    AMDGPUDriver::get_instance().malloc(&ptr, size);
  } else {
    AMDGPUDriver::get_instance().malloc_managed(&ptr, size,
                                                HIP_MEM_ATTACH_GLOBAL);
  }
#else
  TI_NOT_IMPLEMENTED;
#endif

  if (ptr == nullptr) {
    TI_ERROR("Device memory allocation ({} B) failed.", size);
  }

  if (raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is already allocated", ptr);
  }

  raw_memory_chunks_[ptr] = size;
  return ptr;
}

void DeviceMemoryPool::deallocate_raw_memory(void *ptr) {
  /*
    Be aware that this methods is not protected by the mutex.

    deallocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  if (!raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is not allocated", ptr);
  }

#if TI_WITH_CUDA
  CUDADriver::get_instance().mem_free(ptr);
  raw_memory_chunks_.erase(ptr);
#elif TI_WITH_AMDGPU
  AMDGPUDriver::get_instance().mem_free(ptr);
  raw_memory_chunks_.erase(ptr);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void DeviceMemoryPool::reset() {
  std::lock_guard<std::mutex> _(mut_allocation_);

  const auto ptr_map_copied = raw_memory_chunks_;
  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
}

DeviceMemoryPool::~DeviceMemoryPool() {
  reset();
}

const size_t DeviceMemoryPool::page_size{1 << 12};  // 4 KB page size by default

DeviceMemoryPool &DeviceMemoryPool::get_instance() {
  static DeviceMemoryPool *cuda_memory_pool = new DeviceMemoryPool();
  return *cuda_memory_pool;
}

}  // namespace taichi::lang
