#pragma once

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

// A CUDA kernel profiler that uses CUDA timing events
class KernelProfilerCUDA : public KernelProfilerBase {
 public:
  std::string title() const override;
  KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name) override;
  void record(KernelProfilerBase::TaskHandle &task_handle,
              const std::string &task_name) override;
  void sync() override;
  void print() override;
  void clear() override;
  void stop(KernelProfilerBase::TaskHandle handle) override;

 private:
  void *base_event_{nullptr};
  float64 base_time_{0.0};
  std::map<std::string, std::vector<std::pair<void *, void *>>>
      outstanding_events_;
};

TLANG_NAMESPACE_END
