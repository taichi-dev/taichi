#pragma once

#if defined(TI_WITH_CUDA)
#include <taichi/profiler.h>
#include <taichi/platform/cuda/cuda_utils.h>
#include <mutex>

TLANG_NAMESPACE_BEGIN

class CUDAContext {
  CUdevice device;
  CUcontext context;
  int dev_count;
  void *context_buffer;
  std::string mcpu;
  ProfilerBase *profiler;

 public:
  std::mutex lock;

  CUDAContext();

  bool detected() const {
    return dev_count != 0;
  }

  void launch(void *func,
              const std::string &task_name,
              ProfilerBase *profiler,
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
};

// TODO: remove this global var
extern std::unique_ptr<CUDAContext> cuda_context;

TLANG_NAMESPACE_END
#endif
