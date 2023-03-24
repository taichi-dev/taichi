#include "memory_pool.h"
#include "taichi/system/timer.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_device.h"

namespace taichi::lang {

// In the future we wish to move the MemoryPool inside each Device
// so that the memory allocated from each Device can be used as-is.
MemoryPool::MemoryPool(Arch arch, Device *device)
    : arch_(arch), device_(device) {
  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           default_allocator_size / 1024 / 1024);
#if defined(TI_WITH_CUDA)
  // http://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf
  // Stream 0 has special synchronization rules: Operations in stream 0 cannot
  // overlap other streams except for those streams with cudaStreamNonBlocking
  // Do not use cudaCreateStream (with no flags) here!
  if (use_cuda_stream && arch_ == Arch::cuda) {
    CUDADriver::get_instance().stream_create(&cuda_stream,
                                             CU_STREAM_NON_BLOCKING);
  }
#endif
}

void *MemoryPool::allocate(std::size_t size, std::size_t alignment) {
  std::lock_guard<std::mutex> _(mut_allocators);
  void *ret = nullptr;
  if (!allocators.empty()) {
    ret = allocators.back()->allocate(size, alignment);
  }
  if (!ret) {
    // allocation have failed
    auto new_buffer_size = std::max(size, default_allocator_size);
    allocators.emplace_back(
        std::make_unique<UnifiedAllocator>(new_buffer_size, arch_, device_));
    ret = allocators.back()->allocate(size, alignment);
  }
  TI_ASSERT(ret);
  return ret;
}

MemoryPool::~MemoryPool() {
}

}  // namespace taichi::lang
