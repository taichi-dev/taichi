#include "memory_pool.h"
#include <taichi/system/timer.h>
#if TLANG_WITH_CUDA
#include "cuda_runtime.h"
#endif

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool() {
  TC_INFO("Memory pool created. Pre allocator buffer size = {}",
          allocator_size);
  terminating = false;
  killed = false;
  processed_tail = 0;
  th = std::make_unique<std::thread>([this] { this->daemon(); });
}

void MemoryPool::set_queue(MemRequestQueue *queue) {
  std::lock_guard<std::mutex> _(mut);
  this->queue = queue;
}

void MemoryPool::daemon() {
  while (1) {
    Time::usleep(1000);
    std::lock_guard<std::mutex> _(mut);
    if (!queue) {
      continue;
    }
    if (terminating) {
      killed = true;
      break;
    }

    // poll allocation requests.
    using tail_type = decltype(MemRequestQueue::tail);
    tail_type tail;
#if TLANG_WITH_CUDA
    cudaMemcpy(&tail, &queue->tail, sizeof(tail), cudaMemcpyDeviceToHost);
#else
    tail = queue->tail;
#endif
    if (tail > processed_tail) {
      // allocate new buffer
      auto i = processed_tail;
      processed_tail += 1;
      TC_INFO("Processing memory request {}", i);
    }
  }
}

MemoryPool::~MemoryPool() {
  {
    std::lock_guard<std::mutex> _(mut);
    terminating = true;
  }
  th->join();
  TC_ASSERT(killed);
}

TLANG_NAMESPACE_END
