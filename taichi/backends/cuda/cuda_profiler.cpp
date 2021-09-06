#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/system/timeline.h"

TLANG_NAMESPACE_BEGIN

bool KernelProfilerCUDA::is_cuda_profiler(KernelProfilingMode profiling_mode) {
  bool ret = profiling_mode == KernelProfilingMode::enable |
             profiling_mode == KernelProfilingMode::cupti_onepass |
             profiling_mode == KernelProfilingMode::cupti_detailed;
  return ret;
}

bool KernelProfilerCUDA::init_profiler(KernelProfilingMode &profiling_mode) {
  if (!is_cuda_profiler(profiling_mode)) {
    return false;
  }

  KernelProfilingTool profiling_type =
      (profiling_mode == KernelProfilingMode::enable)
          ? KernelProfilingTool::cuevent
          : KernelProfilingTool::cupti;

  if (profiling_type == KernelProfilingTool::cuevent) {
    tool_ = KernelProfilingTool::cuevent;
    mode_ = KernelProfilingMode::enable;
    TI_TRACE("KernelProfilingTool::cuevent : enable");
    TI_TRACE("profiler_type : {}", tool_);
    profiling_mode = mode_;
    return true;
  }

#if !defined(TI_WITH_TOOLKIT_CUDA)
  if (profiling_type == KernelProfilingTool::cupti) {
    TI_WARN(
        "CUPTI toolkit is not compiled with taichi, fallback to cuEvent kernel "
        "profiler");
    TI_WARN(
        "to use CUPTI kernel profiler : "
        "TAICHI_CMAKE_ARGS=-DTI_WITH_TOOLKIT_CUDA=True python3 setup.py "
        "develop --user");

    tool_ = KernelProfilingTool::cuevent;
    mode_ = KernelProfilingMode::enable;

    TI_TRACE("KernelProfilingTool::cuevent : enable");
    TI_TRACE("profiler_type : {}", tool_);
    profiling_mode = mode_;
    return true;
  }
#else
  // TODO::CUPTI_PROFILER
  TI_INFO("TODO::CUPTI_PROFILER");
#endif
}

KernelProfilerCUDA::KernelProfilerCUDA(KernelProfilingMode &mode) {
#if defined(TI_WITH_CUDA)
  this->init_profiler(mode);
  if (tool_ == KernelProfilingTool::cupti) {
    cupti_toolkit_ = std::make_unique<CUPTIToolkit>(mode);
    cupti_toolkit_->init_cupti();
    cupti_toolkit_->begin_profiling();
  }
#endif
  mode_ = mode;
}

void KernelProfilerCUDA::record(KernelProfilerBase::TaskHandle &task_handle,
                                const std::string &task_name) {
  if (tool_ == KernelProfilingTool::cuevent) {
    task_handle = this->start_with_handle(task_name);
  } else if (tool_ == KernelProfilingTool::cupti) {
    cupti_toolkit_->record_launched_kernel(task_name);
  }
}

void KernelProfilerCUDA::clear_toolkit() {
  if (tool_ == KernelProfilingTool::cupti) {
    cupti_toolkit_->clear_traced_records();
  }
}

KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
#if defined(TI_WITH_CUDA)
  void *start, *stop;
  CUDADriver::get_instance().event_create(&start, CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_create(&stop, CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_record(start, 0);
  outstanding_events[kernel_name].push_back(std::make_pair(start, stop));

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
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
#if defined(TI_WITH_CUDA)
  CUDADriver::get_instance().event_record(handle, 0);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::string KernelProfilerCUDA::title() const {
#if defined(TI_WITH_CUDA)
  if (tool_ == KernelProfilingTool::cuevent)
    return "cuEvent Profiler";
  else if (tool_ == KernelProfilingTool::cupti) {
    std::string mode_string = mode_ == KernelProfilingMode::cupti_onepass
                                  ? "accurate mode"
                                  : "detailed mode";
    return "nvCUPTI Profiler :: " + mode_string;
  }
#endif
}

void KernelProfilerCUDA::sync() {
#if defined(TI_WITH_CUDA)
  CUDADriver::get_instance().stream_synchronize(nullptr);

  if (tool_ == KernelProfilingTool::cuevent) {
    auto &timeline = Timeline::get_this_thread_instance();
    for (auto &map_elem : outstanding_events) {
      auto &list = map_elem.second;
      for (auto &item : list) {
        auto start = item.first, stop = item.second;
        float kernel_time;
        CUDADriver::get_instance().event_elapsed_time(&kernel_time, start,
                                                      stop);

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
    outstanding_events.clear();
  } else if (tool_ == KernelProfilingTool::cupti) {
    cupti_toolkit_->calculate_metric_values();
    cupti_toolkit_->statistics_on_traced_records(records, total_time_ms);
    cupti_toolkit_->end_profiling();
    cupti_toolkit_->deinit_cupti();
    cupti_toolkit_->init_cupti();
    cupti_toolkit_->begin_profiling();
  }
#else
  TI_WARN("Profiler not implemented;");
#endif
}

void KernelProfilerCUDA::print() {
#if defined(TI_WITH_CUDA)
  sync();
  if (mode_ != KernelProfilingMode::disable) {
    fmt::print("{}\n", title());
  }
  if (mode_ == KernelProfilingMode::enable) {
    fmt::print(
        "===================================================================="
        "===="
        "=\n");
    fmt::print(
        "[      %     total   count |      min       avg       max   ] "
        "Kernel "
        "name\n");
    std::sort(records.begin(), records.end());
    for (auto &rec : records) {
      auto fraction = rec.total / total_time_ms * 100.0f;
      fmt::print("[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms] {}\n",
                 fraction, rec.total / 1000.0f, rec.counter, rec.min,
                 rec.total / rec.counter, rec.max, rec.name);
    }
    fmt::print(
        "--------------------------------------------------------------------"
        "----"
        "-\n");
    fmt::print(
        "[100.00%] Total kernel execution time: {:7.3f} s   number of "
        "records: "
        "{}\n",
        get_total_time(), records.size());

    fmt::print(
        "===================================================================="
        "===="
        "=\n");
  } else if (mode_ == KernelProfilingMode::cupti_onepass) {
    fmt::print(
        "===================================================================="
        "=============================="
        "=\n");
    fmt::print(
        "[      %     total   count |      min       avg       max    |    "
        "load.g   store.g    ] Kernel "
        "name\n");
    std::sort(records.begin(), records.end());
    for (auto &rec : records) {
      auto fraction = rec.total / total_time_ms * 100.0f;
      fmt::print(
          "[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms | {:9.3f} "
          "{:9.3f}  MB] {}\n",
          fraction, rec.total / 1000.0f, rec.counter, rec.min,
          rec.total / rec.counter, rec.max,
          rec.mem_load_in_bytes / rec.counter / 1024 / 1024,
          rec.mem_store_in_bytes / rec.counter / 1024 / 1024, rec.name);
    }
    fmt::print(
        "--------------------------------------------------------------------"
        "------------------------------"
        "-\n");
    fmt::print(
        "[100.00%] Total kernel execution time: {:7.3f} s   number of "
        "records: "
        "{}\n",
        get_total_time(), records.size());

    fmt::print(
        "===================================================================="
        "=============================="
        "=\n");
  } else if (mode_ == KernelProfilingMode::cupti_detailed) {
    fmt::print(
        "===================================================================="
        "========================================================"
        "=\n");
    fmt::print(
        "[      %     total   count |      min       avg       max    |    "
        "load.g   store.g     |  uti.core   uti.dram ] Kernel "
        "name\n");
    std::sort(records.begin(), records.end());
    for (auto &rec : records) {
      auto fraction = rec.total / total_time_ms * 100.0f;
      fmt::print(
          "[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms | {:9.3f} "
          "{:9.3f}  MB |    {:2.2f}%     {:2.2f}% ] {}\n",
          fraction, rec.total / 1000.0f, rec.counter, rec.min,
          rec.total / rec.counter, rec.max,
          rec.mem_load_in_bytes / rec.counter / 1024 / 1024,
          rec.mem_store_in_bytes / rec.counter / 1024 / 1024,
          rec.utilization_core / rec.counter, rec.utilization_mem / rec.counter,
          rec.name);
    }
    fmt::print(
        "--------------------------------------------------------------------"
        "--------------------------------------------------------"
        "-\n");
    fmt::print(
        "[100.00%] Total kernel execution time: {:7.3f} s   number of "
        "records: "
        "{}\n",
        get_total_time(), records.size());

    fmt::print(
        "===================================================================="
        "========================================================"
        "=\n");
  }
#endif
}

TLANG_NAMESPACE_END
