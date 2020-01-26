#include "memory_pool.h"
#include <taichi/system/timer.h>
#if TLANG_WITH_CUDA
#include "cuda_runtime.h"
#endif

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool() {
  TC_INFO("Memory pool created. Pre allocator buffer size = {}", allocator_size);
  killed = false;
  th = std::make_unique<std::thread>([this]{
    this->daemon();
  });
}

void MemoryPool::daemon() {
  while (1) {
    std::lock_guard<std::mutex> _(mut);
    if (killed) {
      break;
    }

    // poll allocation requests.
#if TLANG_WITH_CUDA
    // cudaMemcpy();
#else
    // memcpy
#endif
    if (true) {
      // allocate new buffer
    }
    Time::usleep(1000);
  }
}

MemoryPool::~MemoryPool() {
  {
    std::lock_guard<std::mutex> _(mut);
    killed = true;
  }
}

TLANG_NAMESPACE_END


