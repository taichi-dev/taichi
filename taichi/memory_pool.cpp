#include "memory_pool.h"
#include <taichi/system/timer.h>
#include "cuda_utils.h"
#if TLANG_WITH_CUDA
#include "cuda_runtime.h"
#include "program.h"

#endif

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool(Program *prog) : prog(prog) {
  TC_INFO("Memory pool created. Pre allocator buffer size = {}",
          allocator_size);
  terminating = false;
  killed = false;
  processed_tail = 0;
  queue = nullptr;
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
    if (prog->config.arch == Arch::gpu) {
#if TLANG_WITH_CUDA
      check_cuda_errors(cudaMemcpy(&tail, &queue->tail, sizeof(tail),
                                   cudaMemcpyDeviceToHost));
#else
      TC_NOT_IMPLEMENTED
#endif
    } else {
      tail = queue->tail;
    }
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
