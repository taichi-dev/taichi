#pragma once

#if defined(TI_WITH_CUDA)
#include <mutex>

#include "taichi/program/profiler.h"
#include "taichi/backends/cuda/cuda_utils.h"

TLANG_NAMESPACE_BEGIN

// Note:
// It would be ideal to create a CUDA context per Taichi program, yet CUDA
// context creation takes time. Therefore we use a shared context to accelerate
// cases such as unit testing where many Taichi programs are created/destroyed.

class CUDAContext {
 private:
  CUdevice device;
  CUcontext context;
  int dev_count;
  std::string mcpu;
  std::mutex lock;
  ProfilerBase *profiler;

  static std::unique_ptr<CUDAContext> instance;

 public:
  CUDAContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();

  bool detected() const {
    return dev_count != 0;
  }

  void launch(void *func,
              const std::string &task_name,
              std::vector<void *> arg_pointers,
              unsigned gridDim,
              unsigned blockDim);

  void set_profiler(ProfilerBase *profiler) {
    this->profiler = profiler;
  }

  std::string get_mcpu() const {
    return mcpu;
  }

  void make_current() {
    check_cuda_error(cuCtxSetCurrent(context));
  }

  ~CUDAContext();

  class ContextGuard {
   private:
    CUcontext old_ctx;

   public:
    ContextGuard(CUDAContext *ctx) {
      check_cuda_error(cuCtxGetCurrent(&old_ctx));
      ctx->make_current();
    }

    ~ContextGuard() {
      check_cuda_error(cuCtxSetCurrent(old_ctx));
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::lock_guard<std::mutex> &&get_lock_guard() {
    return std::move(std::lock_guard<std::mutex>(lock));
  }

  static CUDAContext &get_instance();
};

TLANG_NAMESPACE_END
#endif
