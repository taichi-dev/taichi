#include "memory_pool.h"
#include <taichi/system/timer.h>
#include "cuda_utils.h"
#if TLANG_WITH_CUDA
#include "cuda_runtime.h"
#endif

#include "program.h"

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool(Program *prog) : prog(prog) {
  TC_INFO("Memory pool created. Default buffer size per allocator = {} MB",
          default_allocator_size / 1024 / 1024);
  terminating = false;
  killed = false;
  processed_tail = 0;
  queue = nullptr;
#ifdef TLANG_WITH_CUDA
  // http://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf
  // Stream 0 has special synchronization rules: Operations in stream 0 cannot overlap other streams
  // except for those streams with cudaStreamNonBlocking
  // Do not use cudaCreateStream (with no flags) here!
  check_cuda_errors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
#endif
  th = std::make_unique<std::thread>([this] { this->daemon(); });
}

void MemoryPool::set_queue(MemRequestQueue *queue) {
  std::lock_guard<std::mutex> _(mut);
  this->queue = queue;
}

void *MemoryPool::allocate(std::size_t size, std::size_t alignment) {
  std::lock_guard<std::mutex> _(mut_allocators);
  bool use_cuda = prog->config.arch == Arch::cuda;
  void *ret = nullptr;
  if (!allocators.empty()) {
    ret = allocators.back()->allocate(size, alignment);
  }
  if (!ret) {
    // allocation have failed
    auto new_buffer_size = std::max(size, default_allocator_size);
    allocators.emplace_back(
        std::make_unique<UnifiedAllocator>(new_buffer_size, use_cuda));
    ret = allocators.back()->allocate(size, alignment);
  }
  TC_ASSERT(ret);
  return ret;
}

template <typename T>
T MemoryPool::fetch(volatile void *ptr) {
  T ret;
  if (false && prog->config.arch == Arch::cuda) {
#if TLANG_WITH_CUDA
    check_cuda_errors(cudaMemcpyAsync(&ret, (void *)ptr, sizeof(T),
                                      cudaMemcpyDeviceToHost, cuda_stream));
    check_cuda_errors(cudaStreamSynchronize(cuda_stream));
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    ret = *(T *)ptr;
  }
  return ret;
}

template <typename T>
void MemoryPool::push(volatile T *dest, const T &val) {
  if (false && prog->config.arch == Arch::cuda) {
#if TLANG_WITH_CUDA
    check_cuda_errors(cudaMemcpyAsync((void *)dest, &val, sizeof(T),
                                      cudaMemcpyHostToDevice, cuda_stream));
    check_cuda_errors(cudaStreamSynchronize(cuda_stream));
#else
    TC_NOT_IMPLEMENTED
#endif
  } else {
    *(T *)dest = val;
  }
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
    auto tail = fetch<tail_type>(&queue->tail);
    if (tail > processed_tail) {
      // allocate new buffer
      auto i = processed_tail;
      processed_tail += 1;
      TC_INFO("Processing memory request {}", i);
      auto req = fetch<MemRequest>(&queue->requests[i]);
      TC_INFO("  Allocating memory {} B (alignment {}B) ", req.size,
              req.alignment);
      auto ptr = allocate(req.size, req.alignment);
      TC_INFO("  Allocated. Ptr = {:p}", ptr);
      push(&queue->requests[i].ptr, (uint8 *)ptr);
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
#ifdef TLANG_WITH_CUDA
  check_cuda_errors(cudaStreamDestroy(cuda_stream));
#endif
}

MemoryPool::~MemoryPool() {
  if (!killed) {
    terminate();
  }
}

TLANG_NAMESPACE_END
