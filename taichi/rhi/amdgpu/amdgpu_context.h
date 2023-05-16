#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>

#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"

namespace taichi {
namespace lang {

class AMDGPUDriver;

class AMDGPUContext {
 private:
  void *device_{nullptr};
  void *context_{nullptr};
  int dev_count_{0};
  int compute_capability_{0};
  std::string mcpu_;
  std::mutex lock_;
  KernelProfilerBase *profiler_{nullptr};
  AMDGPUDriver &driver_;
  bool debug_{false};
  std::vector<void *> kernel_arg_pointer_;

 public:
  AMDGPUContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void push_back_kernel_arg_pointer(void *ptr) {
    kernel_arg_pointer_.push_back(ptr);
  }

  void free_kernel_arg_pointer() {
    for (auto &i : kernel_arg_pointer_) {
      AMDGPUDriver::get_instance().mem_free(i);
    }
    kernel_arg_pointer_.erase(kernel_arg_pointer_.begin(),
                              kernel_arg_pointer_.end());
  }

  void pack_args(std::vector<void *> arg_pointers,
                 std::vector<int> arg_sizes,
                 char *arg_packed);

  int get_args_byte(std::vector<int> arg_sizes);

  void set_profiler(KernelProfilerBase *profiler) {
    profiler_ = profiler;
  }

  void launch(void *func,
              const std::string &task_name,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes);

  void set_debug(bool debug) {
    debug_ = debug;
  }

  std::string get_mcpu() const {
    return mcpu_;
  }

  void *get_context() {
    return context_;
  }

  void make_current() {
    driver_.context_set_current(context_);
  }

  int get_compute_capability() const {
    return compute_capability_;
  }

  ~AMDGPUContext();

  class ContextGuard {
   private:
    void *old_ctx_;
    void *new_ctx_;

   public:
    explicit ContextGuard(AMDGPUContext *new_ctx)
        : old_ctx_(nullptr), new_ctx_(new_ctx) {
      AMDGPUDriver::get_instance().context_get_current(&old_ctx_);
      if (old_ctx_ != new_ctx)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_ctx_ != new_ctx_) {
        AMDGPUDriver::get_instance().context_set_current(old_ctx_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return std::unique_lock<std::mutex>(lock_);
  }

  static AMDGPUContext &get_instance();
};

}  // namespace lang
}  // namespace taichi
