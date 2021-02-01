#include "kernel_profiler.h"

#include "taichi/system/timer.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/system/timeline.h"

TLANG_NAMESPACE_BEGIN

void KernelProfileRecord::insert_sample(double t) {
  if (counter == 0) {
    min = t;
    max = t;
  }
  counter++;
  min = std::min(min, t);
  max = std::max(max, t);
  total += t;
}

bool KernelProfileRecord::operator<(const KernelProfileRecord &o) const {
  return total > o.total;
}

void KernelProfilerBase::profiler_start(KernelProfilerBase *profiler,
                                        const char *kernel_name) {
  TI_ASSERT(profiler);
  profiler->start(std::string(kernel_name));
}

void KernelProfilerBase::profiler_stop(KernelProfilerBase *profiler) {
  TI_ASSERT(profiler);
  profiler->stop();
}

void KernelProfilerBase::print() {
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

double KernelProfilerBase::get_total_time() const {
  return total_time_ms / 1000.0;
}

namespace {
// A simple profiler that uses Time::get_time()
class DefaultProfiler : public KernelProfilerBase {
 public:
  explicit DefaultProfiler(Arch arch)
      : title_(fmt::format("{} Profiler", arch_name(arch))) {
  }

  void sync() override {
  }

  std::string title() const override {
    return title_;
  }

  void start(const std::string &kernel_name) override {
    start_t_ = Time::get_time();
    event_name_ = kernel_name;
  }

  void stop() override {
    auto t = Time::get_time() - start_t_;
    auto ms = t * 1000.0;
    auto it = std::find_if(
        records.begin(), records.end(),
        [&](KernelProfileRecord &r) { return r.name == event_name_; });
    if (it == records.end()) {
      records.emplace_back(event_name_);
      it = std::prev(records.end());
    }
    it->insert_sample(ms);
    total_time_ms += ms;
  }

 private:
  double start_t_;
  std::string event_name_;
  std::string title_;
};

// A CUDA kernel profiler that uses CUDA timing events
class KernelProfilerCUDA : public KernelProfilerBase {
 public:
#if defined(TI_WITH_CUDA)

  std::map<std::string, std::vector<std::pair<void *, void *>>>
      outstanding_events;
#endif

  TaskHandle start_with_handle(const std::string &kernel_name) override {
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

  virtual void stop(TaskHandle handle) override {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().event_record(handle, 0);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  std::string title() const override {
    return "CUDA Profiler";
  }

  void sync() override {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
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
#else
    printf("CUDA Profiler not implemented;\n");
#endif
  }

  static KernelProfilerCUDA &get_instance() {
    static KernelProfilerCUDA profiler;
    return profiler;
  }

 private:
  void *base_event_{nullptr};
  float64 base_time_;
};
}  // namespace

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch) {
  if (arch == Arch::cuda) {
    return std::make_unique<KernelProfilerCUDA>();
  } else {
    return std::make_unique<DefaultProfiler>(arch);
  }
}

TLANG_NAMESPACE_END
