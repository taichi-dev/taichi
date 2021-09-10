#include "kernel_profiler.h"

#include "taichi/system/timer.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/system/timeline.h"

TLANG_NAMESPACE_BEGIN

void KernelProfileStatisticalResult::insert_record(double t) {
  if (counter == 0) {
    min = t;
    max = t;
  }
  counter++;
  min = std::min(min, t);
  max = std::max(max, t);
  total += t;
}

bool KernelProfileStatisticalResult::operator<(
    const KernelProfileStatisticalResult &o) const {
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

void KernelProfilerBase::query(const std::string &kernel_name,
                               int &counter,
                               double &min,
                               double &max,
                               double &avg) {
  sync();
  std::regex name_regex(kernel_name + "(.*)");
  for (auto &rec : statistical_results_) {
    if (std::regex_match(rec.name, name_regex)) {
      if (counter == 0) {
        counter = rec.counter;
        min = rec.min;
        max = rec.max;
        avg = rec.total / rec.counter;
      } else if (counter == rec.counter) {
        min += rec.min;
        max += rec.max;
        avg += rec.total / rec.counter;
      } else {
        TI_WARN("{}.counter({}) != {}.counter({}).", kernel_name, counter,
                rec.name, rec.counter);
      }
    }
  }
}

double KernelProfilerBase::get_total_time() const {
  return total_time_ms_ / 1000.0;
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

  void clear() override {
    sync();
    total_time_ms_ = 0;
    traced_records_.clear();
    statistical_results_.clear();
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
    auto it =
        std::find_if(statistical_results_.begin(), statistical_results_.end(),
                     [&](KernelProfileStatisticalResult &r) {
                       return r.name == event_name_;
                     });
    if (it == statistical_results_.end()) {
      statistical_results_.emplace_back(event_name_);
      it = std::prev(statistical_results_.end());
    }
    it->insert_record(ms);
    total_time_ms_ += ms;
  }

 private:
  double start_t_;
  std::string event_name_;
  std::string title_;
};

}  // namespace

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch) {
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<KernelProfilerCUDA>();
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    return std::make_unique<DefaultProfiler>(arch);
  }
}

TLANG_NAMESPACE_END
