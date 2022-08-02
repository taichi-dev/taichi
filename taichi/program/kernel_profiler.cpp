#include "kernel_profiler.h"

#include "taichi/system/timer.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_profiler.h"
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

// TODO : deprecated
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
  void sync() override {
  }

  void update() override {
  }

  void clear() override {
    // sync(); //decoupled: trigger from the foront end
    total_time_ms_ = 0;
    traced_records_.clear();
    statistical_results_.clear();
  }

  void start(const std::string &kernel_name) override {
    start_t_ = Time::get_time();
    event_name_ = kernel_name;
  }

  void stop() override {
    auto t = Time::get_time() - start_t_;
    auto ms = t * 1000.0;
    // trace record
    KernelProfileTracedRecord record;
    record.name = event_name_;
    record.kernel_elapsed_time_in_ms = ms;
    traced_records_.push_back(record);
    // count record
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
};

}  // namespace

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch, bool enable) {
  if (!enable)
    return nullptr;
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<KernelProfilerCUDA>(enable);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    return std::make_unique<DefaultProfiler>();
  }
}

TLANG_NAMESPACE_END
