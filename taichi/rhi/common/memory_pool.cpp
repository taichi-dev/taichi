#include "memory_pool.h"

#include <memory>

// #include "taichi/system/timer.h"

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

class HostMemoryPool : public MemoryPool {
 public:
  HostMemoryPool(Arch arch) {
    arch_ = arch;
    TI_ASSERT(arch_is_cpu(arch_));

    allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator(arch));

    TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
             UnifiedAllocator::default_allocator_size / 1024 / 1024);
  }

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive,
                 bool managed) override {
    std::lock_guard<std::mutex> _(mut_allocation_);

    if (!allocator_) {
      TI_ERROR("Memory pool is already destroyed");
    }
    void *ret = allocator_->allocate(size, alignment, exclusive, managed);
    return ret;
  }

  void release(std::size_t size, void *ptr) override {
    std::lock_guard<std::mutex> _(mut_allocation_);

    if (!allocator_) {
      TI_ERROR("Memory pool is already destroyed");
    }

    if (allocator_->release(size, ptr)) {
      if (dynamic_cast<UnifiedAllocator *>(allocator_.get())) {
        deallocate_raw_memory(ptr);  // release raw memory as well
      }
    }
  }

  void *allocate_raw_memory(std::size_t size, bool managed) override {
    /*
      Be aware that this methods is not protected by the mutex.

      allocate_raw_memory() is designed to be a private method, and
      should only be called by its Allocators friends.

      The caller ensures that no other thread is accessing the memory pool
      when calling this method.
    */

    void *ptr = nullptr;
    if (arch_is_cpu(arch_)) {
#if defined(TI_PLATFORM_UNIX)
      ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      TI_ERROR_IF(ptr == MAP_FAILED, "Virtual memory allocation ({} B) failed.",
                  size);
#else
      MEMORYSTATUSEX stat;
      stat.dwLength = sizeof(stat);
      GlobalMemoryStatusEx(&stat);
      if (stat.ullAvailVirtual < size) {
        TI_P(stat.ullAvailVirtual);
        TI_P(size);
        TI_ERROR("Insufficient virtual memory space");
      }
      ptr =
          VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
      TI_ERROR_IF(ptr == nullptr, "Virtual memory allocation ({} B) failed.",
                  size);
#endif
      TI_ERROR_IF(((uint64_t)ptr) % page_size != 0,
                  "Allocated address ({:}) is not aligned by page size {}", ptr,
                  page_size);
    } else {
      TI_NOT_IMPLEMENTED;
    }

    if (raw_memory_chunks_.count(ptr)) {
      TI_ERROR("Memory address ({:}) is already allocated", ptr);
    }

    raw_memory_chunks_[ptr] = size;
    return ptr;
  }

  void deallocate_raw_memory(void *ptr) override {
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

    std::size_t size = raw_memory_chunks_[ptr];
#if defined(TI_PLATFORM_UNIX)
    if (munmap(ptr, size) != 0)
#else
    // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree
    // According to MS Doc: size must be when using MEM_RELEASE
    if (!VirtualFree(ptr, 0, MEM_RELEASE))
#endif
      TI_ERROR("Failed to free virtual memory ({} B)", size);

    raw_memory_chunks_.erase(ptr);
  }

  void reset() override {
    std::lock_guard<std::mutex> _(mut_allocation_);
    allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator(arch_));

    const auto ptr_map_copied = raw_memory_chunks_;
    for (auto &ptr : ptr_map_copied) {
      deallocate_raw_memory(ptr.first);
    }
  }

  ~HostMemoryPool() {
    reset();
  }
};

class CudaMemoryPool : public MemoryPool {
 public:
  CudaMemoryPool(Arch arch) {
    arch_ = arch;
    TI_ASSERT(arch_is_cuda(arch_));

    // TODO(zhanlue): replace UnifiedAllocator with CachingAllocator
    allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator(arch));
  }

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive,
                 bool managed) override {
    std::lock_guard<std::mutex> _(mut_allocation_);

    // TODO(zhanlue): Replace UnifiedAllocator with Caching Allocator
    // Here we reuse the UnifiedAllocator's allocation logic to perform
    // pre-allocation, but ideally this preallocation logic should be fused into
    // the Caching Allocator
    if (!allocator_) {
      TI_ERROR("Memory pool is already destroyed");
    }

    void *ret = allocator_->allocate(size, alignment, exclusive, managed);
    return ret;
  }

  void release(std::size_t size, void *ptr) override {
    std::lock_guard<std::mutex> _(mut_allocation_);

    if (!allocator_) {
      TI_ERROR("Memory pool is already destroyed");
    }

    if (allocator_->release(size, ptr)) {
      if (dynamic_cast<UnifiedAllocator *>(allocator_.get())) {
        deallocate_raw_memory(ptr);  // release raw memory as well
      }
    }
  }

  void *allocate_raw_memory(std::size_t size, bool managed) override {
    /*
      Be aware that this methods is not protected by the mutex.

      allocate_raw_memory() is designed to be a private method, and
      should only be called by its Allocators friends.

      The caller ensures that no other thread is accessing the memory pool
      when calling this method.
    */
#ifdef TI_WITH_CUDA
    void *ptr = nullptr;
    if (!managed) {
      CUDADriver::get_instance().malloc(&ptr, size);
    } else {
      CUDADriver::get_instance().malloc_managed(&ptr, size,
                                                CU_MEM_ATTACH_GLOBAL);
    }

    if (ptr == nullptr) {
      TI_ERROR("CUDA memory allocation ({} B) failed.", size);
    }

    if (raw_memory_chunks_.count(ptr)) {
      TI_ERROR("Memory address ({:}) is already allocated", ptr);
    }

    raw_memory_chunks_[ptr] = size;
    return ptr;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  void deallocate_raw_memory(void *ptr) override {
    /*
      Be aware that this methods is not protected by the mutex.

      deallocate_raw_memory() is designed to be a private method, and
      should only be called by its Allocators friends.

      The caller ensures that no other thread is accessing the memory pool
      when calling this method.
    */
#ifdef TI_WITH_CUDA
    if (!raw_memory_chunks_.count(ptr)) {
      TI_ERROR("Memory address ({:}) is not allocated", ptr);
    }
    CUDADriver::get_instance().mem_free(ptr);
    raw_memory_chunks_.erase(ptr);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  void reset() override {
    std::lock_guard<std::mutex> _(mut_allocation_);

    const auto ptr_map_copied = raw_memory_chunks_;
    for (auto &ptr : ptr_map_copied) {
      deallocate_raw_memory(ptr.first);
    }
  }

  ~CudaMemoryPool() {
    reset();
  }
};

const size_t MemoryPool::page_size{1 << 12};  // 4 KB page size by default

MemoryPool &MemoryPool::get_instance(Arch arch) {
  if (!arch_is_cuda(arch) && !arch_is_cpu(arch)) {
    arch = host_arch();
  }

  if (arch_is_cuda(arch)) {
    static MemoryPool *cuda_memory_pool = new CudaMemoryPool(arch);
    return *cuda_memory_pool;
  }

  static MemoryPool *cpu_memory_pool = new HostMemoryPool(arch);
  return *cpu_memory_pool;
}

}  // namespace taichi::lang
