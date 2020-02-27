#include "profiler.h"

#include <taichi/system/timer.h>

#if defined(TI_WITH_CUDA)
#include <cuda_runtime.h>
#endif

TLANG_NAMESPACE_BEGIN

void ProfileRecord::insert_sample(double t) {
  if (counter == 0) {
    min = t;
    max = t;
  }
  counter++;
  min = std::min(min, t);
  max = std::max(max, t);
  total += t;
}

void ProfilerBase::profiler_start(ProfilerBase *profiler,
                                  const char *kernel_name) {
  profiler->start(std::string(kernel_name));
}

void ProfilerBase::profiler_stop(ProfilerBase *profiler) {
  profiler->stop();
}

void ProfilerBase::print() {
  sync();
  printf("%s\n", title().c_str());
  for (auto &rec : records) {
    printf(
        "[%6.2f%%] %-40s     min %7.3f ms   avg %7.3f ms    max %7.3f ms   "
        "total %7.3f s [%7dx]\n",
        rec.total / total_time * 100.0f, rec.name.c_str(), rec.min,
        rec.total / rec.counter, rec.max, rec.total / 1000.0f, rec.counter);
  }
}

namespace {
// A simple profiler that uses Time::get_time()
class DefaultProfiler : public ProfilerBase {
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
    auto it =
        std::find_if(records.begin(), records.end(),
                     [&](ProfileRecord &r) { return r.name == event_name_; });
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
class CUDAProfiler : public ProfilerBase {
 public:
#if defined(TI_WITH_CUDA)
  cudaEvent_t current_stop;

  std::map<std::string, std::vector<std::pair<cudaEvent_t, cudaEvent_t>>>
      outstanding_events;
#endif

  void start(const std::string &kernel_name) override {
#if defined(TI_WITH_CUDA)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    outstanding_events[kernel_name].push_back(std::make_pair(start, stop));
    current_stop = stop;
#else
    printf("CUDA Profiler not implemented;\n");
#endif
  }

  virtual void stop() override {
#if defined(TI_WITH_CUDA)
    cudaEventRecord(current_stop);
#else
    printf("CUDA Profiler not implemented;\n");
#endif
  }

  std::string title() const override {
    return "CUDA Profiler";
  }

  void sync() override {
#if defined(TI_WITH_CUDA)
    cudaDeviceSynchronize();
    for (auto &map_elem : outstanding_events) {
      auto &list = map_elem.second;
      for (auto &item : list) {
        auto start = item.first, stop = item.second;
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        auto it = std::find_if(
            records.begin(), records.end(),
            [&](ProfileRecord &r) { return r.name == map_elem.first; });
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
    printf("GPU Profiler not implemented;\n");
#endif
  }

  static CUDAProfiler &get_instance() {
    static CUDAProfiler profiler;
    return profiler;
  }
};
}  // namespace

std::unique_ptr<ProfilerBase> make_profiler(Arch arch) {
  if (arch == Arch::x64 || arch == Arch::arm64 || arch == Arch::metal || arch == Arch::opengl) {
    return std::make_unique<DefaultProfiler>(arch);
  } else if (arch == Arch::cuda) {
    return std::make_unique<CUDAProfiler>();
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

TLANG_NAMESPACE_END
