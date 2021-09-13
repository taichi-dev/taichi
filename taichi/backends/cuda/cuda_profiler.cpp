#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)

bool check_device_capability() {
  void *device;
  int cc_major;
  CUDADriver::get_instance().device_get(&device, 0);
  CUDADriver::get_instance().device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (cc_major < 7) {
    TI_WARN(
        "CUPTI profiler APIs unsupported on Device with compute capability < "
        "7.0 , fallback to default kernel profiler");
    return false;
  }
  return true;
}

// The init logic here is temporarily set up for test CUPTI
// will not affect default toolkit (cuEvent)
KernelProfilerCUDA::KernelProfilerCUDA() {
// if Taichi was compiled with CUDA toolit, then use CUPTI
// TODO : add set_mode() to select toolkit by user
#if defined(TI_WITH_CUDA_TOOLKIT)
  if (check_device_capability())
    tool_ = ProfilingToolkit::cupti;
#endif

  if (tool_ == ProfilingToolkit::event) {
    event_toolkit_ = std::make_unique<EventToolkit>();
  } else if (tool_ == ProfilingToolkit::cupti) {
    cupti_toolkit_ = std::make_unique<CuptiToolkit>();
  }
}

// deprecated, move to trace()
KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}

void KernelProfilerCUDA::trace(KernelProfilerBase::TaskHandle &task_handle,
                               const std::string &task_name) {
  if (tool_ == ProfilingToolkit::event)
    task_handle = event_toolkit_->start_with_handle(task_name);
  else if (tool_ == ProfilingToolkit::cupti) {
    TI_NOT_IMPLEMENTED;
  }
}

void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
  if (tool_ == ProfilingToolkit::event)
    CUDADriver::get_instance().event_record(handle, 0);
}

bool KernelProfilerCUDA::statistics_on_traced_records() {
  for (auto &record : traced_records_) {
    auto it =
        std::find_if(statistical_results_.begin(), statistical_results_.end(),
                     [&](KernelProfileStatisticalResult &result) {
                       return result.name == record.name;
                     });
    if (it == statistical_results_.end()) {
      statistical_results_.emplace_back(record.name);
      it = std::prev(statistical_results_.end());
    }
    it->insert_record(record.kernel_elapsed_time_in_ms);
    total_time_ms_ += record.kernel_elapsed_time_in_ms;
  }

  return true;
}

void KernelProfilerCUDA::sync() {
  // sync
  CUDADriver::get_instance().stream_synchronize(nullptr);

  // update
  if (tool_ == ProfilingToolkit::event) {
    event_toolkit_->update_record(traced_records_);
    event_toolkit_->update_timeline(traced_records_);
    statistics_on_traced_records();
    event_toolkit_->clear();
  } else if (tool_ == ProfilingToolkit::cupti) {
    cupti_toolkit_->calculate_metric_values();
    statistics_on_traced_records();
    cupti_toolkit_->end_profiling();
    cupti_toolkit_->deinit_cupti();
    // reinit is necessary to get accurate  result
    cupti_toolkit_->init_cupti();  // data image will be cleared
    cupti_toolkit_->begin_profiling();
  }
}

void KernelProfilerCUDA::print() {
  sync();
  fmt::print("{}\n", title());
  fmt::print(
      "========================================================================"
      "=\n");
  fmt::print(
      "[      %     total   count |      min       avg       max   ] Kernel "
      "name\n");
  std::sort(statistical_results_.begin(), statistical_results_.end());
  for (auto &rec : statistical_results_) {
    auto fraction = rec.total / total_time_ms_ * 100.0f;
    fmt::print("[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms] {}\n",
               fraction, rec.total / 1000.0f, rec.counter, rec.min,
               rec.total / rec.counter, rec.max, rec.name);
  }
  fmt::print(
      "------------------------------------------------------------------------"
      "-\n");
  fmt::print(
      "[100.00%] Total kernel execution time: {:7.3f} s   number of records: "
      "{}\n",
      get_total_time(), statistical_results_.size());

  fmt::print(
      "========================================================================"
      "=\n");
}

void KernelProfilerCUDA::clear() {
  sync();
  total_time_ms_ = 0;
  traced_records_.clear();
  statistical_results_.clear();
}

#else

KernelProfilerCUDA::KernelProfilerCUDA() {
  TI_NOT_IMPLEMENTED;
}
KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::trace(KernelProfilerBase::TaskHandle &task_handle,
                               const std::string &task_name) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::sync() {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::clear() {
  TI_NOT_IMPLEMENTED;
}

#endif

// default profiling toolkit : cuEvent
// for now put it together with KernelProfilerCUDA
#if defined(TI_WITH_CUDA)
KernelProfilerBase::TaskHandle EventToolkit::start_with_handle(
    const std::string &kernel_name) {
  EventRecord record;
  record.name = kernel_name;

  CUDADriver::get_instance().event_create(&(record.start_event),
                                          CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_create(&(record.stop_event),
                                          CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_record((record.start_event), 0);
  event_records_.push_back(record);

  if (!base_event_) {
    // Note that CUDA driver API only allows querying relative time difference
    // between two events, therefore we need to manually build a mapping
    // between GPU and CPU time.
    // TODO: periodically reinitialize for more accuracy.
    int n_iters = 100;
    // Warm up CUDA driver, and use the final event as the base event.
    for (int i = 0; i < n_iters; i++) {
      void *e;
      CUDADriver::get_instance().event_create(&e, CU_EVENT_DEFAULT);
      CUDADriver::get_instance().event_record(e, 0);
      CUDADriver::get_instance().event_synchronize(e);
      auto final_t = Time::get_time();
      if (i == n_iters - 1) {
        base_event_ = e;
        // TODO: figure out a better way to synchronize CPU and GPU time.
        constexpr float64 cuda_time_offset = 3e-4;
        // Since event recording and synchronization can take 5 us, it's hard
        // to exactly measure the real event time. Also note there seems to be
        // a systematic time offset on CUDA, so adjust for that using a 3e-4 s
        // magic number.
        base_time_ = final_t + cuda_time_offset;
      } else {
        CUDADriver::get_instance().event_destroy(e);
      }
    }
  }

  return record.stop_event;
}

void EventToolkit::update_record(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  // cuEvent : get kernel_elapsed_time
  for (auto &record : event_records_) {
    CUDADriver::get_instance().event_elapsed_time(
        &record.kernel_elapsed_time_in_ms, record.start_event,
        record.stop_event);
    CUDADriver::get_instance().event_elapsed_time(
        &record.time_since_base, base_event_, record.start_event);

    // TODO: the following two lines seem to increases profiler overhead a
    // little bit. Is there a way to avoid the overhead while not creating
    // too many events?
    CUDADriver::get_instance().event_destroy(record.start_event);
    CUDADriver::get_instance().event_destroy(record.stop_event);

    // copy to traced_records_ then clear event_records_
    KernelProfileTracedRecord traced_record;
    traced_record.name = record.name;
    traced_record.kernel_elapsed_time_in_ms = record.kernel_elapsed_time_in_ms;
    traced_record.time_since_base = record.time_since_base;
    traced_records.push_back(traced_record);
  }
}

void EventToolkit::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  if (Timelines::get_instance().get_enabled()) {
    auto &timeline = Timeline::get_this_thread_instance();
    for (auto &record : traced_records) {
      // param of insert_event() :
      // struct TimelineEvent @ taichi/taichi/system/timeline.h
      timeline.insert_event({record.name, /*param_name=begin*/ true,
                             base_time_ + record.time_since_base * 1e-3,
                             "cuda"});
      timeline.insert_event({record.name, /*param_name=begin*/ false,
                             base_time_ + (record.time_since_base +
                                           record.kernel_elapsed_time_in_ms) *
                                              1e-3,
                             "cuda"});
    }
  }
}

#else
KernelProfilerBase::TaskHandle EventToolkit::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkit::update_record(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkit::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
#endif

TLANG_NAMESPACE_END
