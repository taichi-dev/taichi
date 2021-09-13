#pragma once

#include "taichi/system/timeline.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/cuda/cupti_toolkit.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

enum class ProfilingToolkit : int {
  event,
  cupti,
};

class EventToolkit;

// A CUDA kernel profiler
class KernelProfilerCUDA : public KernelProfilerBase {
 public:
  KernelProfilerCUDA();
  std::string title() const override {
    return "CUDA Profiler";
  }

  void trace(KernelProfilerBase::TaskHandle &task_handle,
             const std::string &task_name) override;
  void sync() override;
  void print() override;
  void clear() override;
  void stop(KernelProfilerBase::TaskHandle handle) override;

  bool statistics_on_traced_records();

  KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name) override;

 private:
  ProfilingToolkit tool_ = ProfilingToolkit::event;
  std::unique_ptr<EventToolkit> event_toolkit_{nullptr};
  // if(tool_ == ProfilingToolkit::cupti) event_toolkit_ = nullptr
  std::unique_ptr<CuptiToolkit> cupti_toolkit_{nullptr};
  // if(tool_ == ProfilingToolkit::event) cupti_toolkit_ = nullptr
  // TODO : switch profiling toolkit at runtime
};

// default profiling toolkit
class EventToolkit {
 public:
  void update_record(std::vector<KernelProfileTracedRecord> &traced_records);
  KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name);
  void update_timeline(std::vector<KernelProfileTracedRecord> &traced_records);
  void clear() {
    event_records_.clear();
  }

 private:
  struct EventRecord {
    std::string name;
    float kernel_elapsed_time_in_ms{0.0};
    float time_since_base{0.0};
    void *start_event{nullptr};
    void *stop_event{nullptr};
  };
  float64 base_time_{0.0};
  void *base_event_{nullptr};
  // for cuEvent profiling, clear after sync()
  std::vector<EventRecord> event_records_;
};

TLANG_NAMESPACE_END
