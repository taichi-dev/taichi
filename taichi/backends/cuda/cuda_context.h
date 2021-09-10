#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"

TLANG_NAMESPACE_BEGIN

// Note:
// It would be ideal to create a CUDA context per Taichi program, yet CUDA
// context creation takes time. Therefore we use a shared context to accelerate
// cases such as unit testing where many Taichi programs are created/destroyed.

class CUDADriver;

class CUDAContext {
 private:
  void *device;
  void *context;
  int dev_count;
  int compute_capability;
  std::string mcpu;
  std::mutex lock;
  KernelProfilerBase *profiler;
  CUDADriver &driver;
  bool debug;

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
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t shared_mem_bytes);

  void set_profiler(KernelProfilerBase *profiler) {
    this->profiler = profiler;
  }

  void set_debug(bool debug_) {
    debug = debug_;
  }

  std::string get_mcpu() const {
    return mcpu;
  }

  void make_current() {
    driver.context_set_current(context);
  }

  int get_compute_capability() const {
    return compute_capability;
  }

  ~CUDAContext();

  class ContextGuard {
   private:
    void *old_ctx;
    void *new_ctx;

   public:
    ContextGuard(CUDAContext *new_ctx) : old_ctx(nullptr), new_ctx(new_ctx) {
      CUDADriver::get_instance().context_get_current(&old_ctx);
      if (old_ctx != new_ctx)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_ctx != new_ctx) {
        CUDADriver::get_instance().context_set_current(old_ctx);
      }
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
