#pragma once

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/cuda/cupti_toolkit.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

// A CUDA kernel profiler that uses CUDA timing events
class KernelProfilerCUDA : public KernelProfilerBase {
 public:
#if defined(TI_WITH_CUDA)
  std::map<std::string, std::vector<std::pair<void *, void *>>>
      outstanding_events;
#endif

  explicit KernelProfilerCUDA(KernelProfilingMode &mode);

  bool is_cuda_profiler(KernelProfilingMode profiling_mode);
  bool init_profiler(KernelProfilingMode &profiling_mode);

  KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name) override;

  void record(KernelProfilerBase::TaskHandle &task_handle,
              const std::string &task_name) override;
  void stop(KernelProfilerBase::TaskHandle handle) override;
  std::string title() const override;
  void sync() override;
  void print() override;
  void clear_toolkit() override;

 private:
  void *base_event_{nullptr};
  float64 base_time_{0.0};
  std::unique_ptr<CUPTIToolkit> cupti_toolkit_{nullptr};
};

TLANG_NAMESPACE_END
