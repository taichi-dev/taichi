#pragma once

#include "taichi/system/timeline.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cupti_toolkit.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

enum class ProfilingToolkit : int {
  undef,
  event,
  cupti,
};

class EventToolkit;

// A CUDA kernel profiler
class KernelProfilerCUDA : public KernelProfilerBase {
 public:
  KernelProfilerCUDA(bool enable);

  std::string get_device_name() override;

  bool reinit_with_metrics(const std::vector<std::string> metrics) override;
  void trace(KernelProfilerBase::TaskHandle &task_handle,
             const std::string &kernel_name,
             void *kernel,
             uint32_t grid_size,
             uint32_t block_size,
             uint32_t dynamic_smem_size);
  void sync() override;
  void update() override;
  void clear() override;
  void stop(KernelProfilerBase::TaskHandle handle) override;

  bool set_profiler_toolkit(std::string toolkit_name) override;

  bool statistics_on_traced_records();

  KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name) override;

  bool record_kernel_attributes(void *kernel,
                                uint32_t grid_size,
                                uint32_t block_size,
                                uint32_t dynamic_smem_size);

 private:
  ProfilingToolkit tool_ = ProfilingToolkit::undef;

  // Instances of these toolkits may exist at the same time,
  // but only one will be enabled.
  std::unique_ptr<EventToolkit> event_toolkit_{nullptr};
  std::unique_ptr<CuptiToolkit> cupti_toolkit_{nullptr};
  std::vector<std::string> metric_list_;
  uint32_t records_size_after_sync_{0};
};

// default profiling toolkit
class EventToolkit {
 public:
  void update_record(uint32_t records_size_after_sync,
                     std::vector<KernelProfileTracedRecord> &traced_records);
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

 public:
  EventRecord *get_current_event_record() {
    return &(event_records_.back());
  }
  void *get_base_event() const {
    return base_event_;
  }
};

TLANG_NAMESPACE_END
