#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/system/timeline.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)

std::string KernelProfilerCUDA::title() const {
  return "CUDA Profiler";
}

KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  void *start, *stop;
  CUDADriver::get_instance().event_create(&start, CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_create(&stop, CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_record(start, 0);
  outstanding_events_[kernel_name].push_back(std::make_pair(start, stop));

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

  return stop;
}

void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
  CUDADriver::get_instance().event_record(handle, 0);
}

void KernelProfilerCUDA::record(KernelProfilerBase::TaskHandle &task_handle,
                                const std::string &task_name) {
  task_handle = this->start_with_handle(task_name);
}

void KernelProfilerCUDA::sync() {
  CUDADriver::get_instance().stream_synchronize(nullptr);
  auto &timeline = Timeline::get_this_thread_instance();
  for (auto &map_elem : outstanding_events_) {
    auto &list = map_elem.second;
    for (auto &item : list) {
      auto start = item.first, stop = item.second;
      float kernel_time;
      CUDADriver::get_instance().event_elapsed_time(&kernel_time, start, stop);

      if (Timelines::get_instance().get_enabled()) {
        float time_since_base;
        CUDADriver::get_instance().event_elapsed_time(&time_since_base,
                                                      base_event_, start);
        timeline.insert_event({map_elem.first, true,
                               base_time_ + time_since_base * 1e-3, "cuda"});
        timeline.insert_event(
            {map_elem.first, false,
             base_time_ + (time_since_base + kernel_time) * 1e-3, "cuda"});
      }

      auto it = std::find_if(
          records.begin(), records.end(),
          [&](KernelProfileRecord &r) { return r.name == map_elem.first; });
      if (it == records.end()) {
        records.emplace_back(map_elem.first);
        it = std::prev(records.end());
      }
      it->insert_sample(kernel_time);
      total_time_ms += kernel_time;

      // TODO: the following two lines seem to increases profiler overhead a
      // little bit. Is there a way to avoid the overhead while not creating
      // too many events?
      CUDADriver::get_instance().event_destroy(start);
      CUDADriver::get_instance().event_destroy(stop);
    }
  }
  outstanding_events_.clear();
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
  std::sort(records.begin(), records.end());
  for (auto &rec : records) {
    auto fraction = rec.total / total_time_ms * 100.0f;
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
      get_total_time(), records.size());

  fmt::print(
      "========================================================================"
      "=\n");
}

void KernelProfilerCUDA::clear() {
  sync();
  total_time_ms = 0;
  records.clear();
}

#else

std::string KernelProfilerCUDA::title() const {
  TI_NOT_IMPLEMENTED;
}
KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::record(KernelProfilerBase::TaskHandle &task_handle,
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

TLANG_NAMESPACE_END
