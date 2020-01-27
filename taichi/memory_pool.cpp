#include "memory_pool.h"
#include <taichi/system/timer.h>
#include "cuda_utils.h"
#if TLANG_WITH_CUDA
#include "cuda_runtime.h"
#include "program.h"

#endif

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool(Program *prog) : prog(prog) {
  TC_INFO("Memory pool created. Default buffer size per allocator = {} MB",
          default_allocator_size / 1024 / 1024);
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

void *MemoryPool::allocate(std::size_t size, std::size_t alignment) {
  std::lock_guard<std::mutex> _(mut);
  bool use_cuda = prog->config.arch == Arch::cuda;
  if (allocators.empty()) {
    allocators.emplace_back(std::make_unique<UnifiedAllocator>(use_cuda));
  }
  auto ret = allocators.back()->allocate(size, alignment);
  TC_ASSERT(ret);
  return ret;
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
    if (prog->config.arch == Arch::cuda) {
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

void MemoryPool::terminate() {
  {
    std::lock_guard<std::mutex> _(mut);
    terminating = true;
  }
  th->join();
  TC_ASSERT(killed);
}

MemoryPool::~MemoryPool() {
  if (!killed) {
    terminate();
  }
}

TLANG_NAMESPACE_END
