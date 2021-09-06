#include "kernel_profiler.h"

#include "taichi/system/timer.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/system/timeline.h"

TLANG_NAMESPACE_BEGIN

std::string kernel_profiler_name(KernelProfilingMode mode) {
  switch (mode) {
#define PER_MODE(x)            \
  case KernelProfilingMode::x: \
    return #x;                 \
    break;
#include "taichi/inc/kernel_profiling_mode.inc.h"

#undef PER_MODE
    default:
      TI_NOT_IMPLEMENTED
  }
}

KernelProfilingMode kernel_profiler_from_name(const std::string &mode) {
#define PER_MODE(x)                \
  else if (mode == #x) {           \
    return KernelProfilingMode::x; \
  }

  if (false) {
  }
#include "taichi/inc/kernel_profiling_mode.inc.h"

  else {
    TI_ERROR("Unknown kernel profiler name: {}", mode);
  }

#undef PER_MODE
}

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

void KernelProfileRecord::cuda_mem_access(float load, float store) {
  mem_load_in_bytes += load;
  mem_store_in_bytes += store;
}

void KernelProfileRecord::cuda_utilization_ratio(float core, float mem) {
  utilization_core += core;
  utilization_mem += mem;
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

void KernelProfilerBase::query(const std::string &kernel_name,
                               int &counter,
                               double &min,
                               double &max,
                               double &avg) {
  sync();
  std::regex name_regex(kernel_name + "(.*)");
  for (auto &rec : records) {
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
  return total_time_ms / 1000.0;
}

void KernelProfilerBase::clear() {
  sync();
  total_time_ms = 0;
  records.clear();
  clear_toolkit();
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

}  // namespace

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch,
                                                  KernelProfilingMode &mode) {
  if (arch == Arch::cuda) {
    return std::make_unique<KernelProfilerCUDA>(mode);
  } else {
    return std::make_unique<DefaultProfiler>(arch);
  }
}

TLANG_NAMESPACE_END
