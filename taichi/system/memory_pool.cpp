#include "memory_pool.h"
#include "taichi/system/timer.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

MemoryPool::MemoryPool(Program *prog) : prog(prog) {
  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           default_allocator_size / 1024 / 1024);
  terminating = false;
  killed = false;
  use_unified_memory = prog->config.use_unified_memory;
  processed_tail = 0;
  queue = nullptr;
#if 0 && defined(TI_WITH_CUDA)
  // http://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf
  // Stream 0 has special synchronization rules: Operations in stream 0 cannot
  // overlap other streams except for those streams with cudaStreamNonBlocking
  // Do not use cudaCreateStream (with no flags) here!
  if (prog->config.arch == Arch::cuda) {
    check_cuda_error(
        cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
  }
#endif
  th = std::make_unique<std::thread>([this] { this->daemon(); });
}

void MemoryPool::set_queue(MemRequestQueue *queue) {
  std::lock_guard<std::mutex> _(mut);
  this->queue = queue;
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
        std::make_unique<UnifiedAllocator>(new_buffer_size, prog->config.arch));
    ret = allocators.back()->allocate(size, alignment);
  }
  TI_ASSERT(ret);
  return ret;
}

template <typename T>
T MemoryPool::fetch(volatile void *ptr) {
  T ret;
  /*
  if (false && prog->config.arch == Arch::cuda) {
#if TI_WITH_CUDA
    CUDADriver::get_instance().cudaMemcpyAsync(&ret, (void *)ptr, sizeof(T,
                                     cudaMemcpyDeviceToHost, cuda_stream));
    CUDADriver::get_instance().cudaStreamSynchronize(cuda_stream);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
  */
  ret = *(T *)ptr;
  //}
  return ret;
}

template <typename T>
void MemoryPool::push(volatile T *dest, const T &val) {
  /*
  if (false && prog->config.arch == Arch::cuda) {
#if TI_WITH_CUDA
    CUDADriver::get_instance().cudaMemcpyAsync((void *)dest, &val, sizeof(T,
                                     cudaMemcpyHostToDevice, cuda_stream));
    CUDADriver::get_instance().cudaStreamSynchronize(cuda_stream);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
  */
  *(T *)dest = val;
  // }
}

void MemoryPool::daemon() {
  while (1) {
    Time::usleep(1000);
    std::lock_guard<std::mutex> _(mut);
    if (terminating) {
      killed = true;
      break;
    }
    if (!queue) {
      continue;
    }

    // poll allocation requests.
    using tail_type = decltype(MemRequestQueue::tail);
    auto tail = fetch<tail_type>(&queue->tail);
    if (tail > processed_tail) {
      // allocate new buffer
      auto i = processed_tail;
      TI_DEBUG("Processing memory alloc request {}", i);
      auto req = fetch<MemRequest>(&queue->requests[i]);
      if (req.size == 0 || req.alignment == 0) {
        TI_DEBUG(" Incomplete memory alloc request {} fetched. Skipping", i);
        continue;
      }
      TI_DEBUG("  Allocating memory {} B (alignment {}B) ", req.size,
               req.alignment);
      auto ptr = allocate(req.size, req.alignment);
      TI_DEBUG("  Allocated. Ptr = {:p}", ptr);
      push(&queue->requests[i].ptr, (uint8 *)ptr);
      processed_tail += 1;
    }
  }
}

void MemoryPool::terminate() {
  {
    std::lock_guard<std::mutex> _(mut);
    terminating = true;
  }
  th->join();
  TI_ASSERT(killed);
#if 0 && defined(TI_WITH_CUDA)
  if (prog->config.arch == Arch::cuda)
    CUDADriver::get_instance().cudaStreamDestroy(cuda_stream);
#endif
}

MemoryPool::~MemoryPool() {
  if (!killed) {
    terminate();
  }
}

TLANG_NAMESPACE_END
