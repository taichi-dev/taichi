#include "kernel_profiler.h"

#include "taichi/system/timer.h"
#include "taichi/backends/cuda/cuda_driver.h"

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
    auto fraction = rec.total / total_time * 100.0f;
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
      total_time / 1000.0f, records.size());
  fmt::print(
      "========================================================================"
      "=\n");
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
    total_time += ms;
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
  void *current_stop;

  std::map<std::string, std::vector<std::pair<void *, void *>>>
      outstanding_events;
#endif

  void start(const std::string &kernel_name) override {
#if defined(TI_WITH_CUDA)
    void *start, *stop;
    CUDADriver::get_instance().event_create(&start, CU_EVENT_DEFAULT);
    CUDADriver::get_instance().event_create(&stop, CU_EVENT_DEFAULT);
    CUDADriver::get_instance().event_record(start, 0);
    outstanding_events[kernel_name].push_back(std::make_pair(start, stop));
    current_stop = stop;
#else
    printf("CUDA Profiler not implemented;\n");
#endif
  }

  virtual void stop() override {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().event_record(current_stop, 0);
#else
    printf("CUDA Profiler not implemented;\n");
#endif
  }

  std::string title() const override {
    return "CUDA Profiler";
  }

  void sync() override {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto &map_elem : outstanding_events) {
      auto &list = map_elem.second;
      for (auto &item : list) {
        auto start = item.first, stop = item.second;
        float ms;
        CUDADriver::get_instance().event_elapsed_time(&ms, start, stop);
        auto it = std::find_if(
            records.begin(), records.end(),
            [&](KernelProfileRecord &r) { return r.name == map_elem.first; });
        if (it == records.end()) {
          records.emplace_back(map_elem.first);
          it = std::prev(records.end());
        }
        it->insert_sample(ms);
        total_time += ms;
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
