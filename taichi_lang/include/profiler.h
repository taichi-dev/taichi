#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include "common.h"

TLANG_NAMESPACE_BEGIN

struct ProfileRecord {
  std::string name;
  int counter;
  double min;
  double max;
  double total;

  ProfileRecord(const std::string &name)
      : name(name), counter(0), min(0), max(0), total(0) {
  }

  void insert_sample(double t) {
    if (counter == 0) {
      min = t;
      max = t;
    }
    counter++;
    min = std::min(min, t);
    max = std::max(max, t);
    total += t;
  }
};

class ProfilerBase {
 protected:
  std::vector<ProfileRecord> records;
  double total_time;

 public:
  void clear() {
    total_time = 0;
    records.clear();
  }

  virtual void sync() = 0;

  virtual std::string title() = 0;

  void print() {
    sync();
    printf("%s\n", title().c_str());
    for (auto &rec : records) {
      printf(
          "[%6.2f%%] %30s     min %7.3f ms   avg %7.3f ms    max %7.3f ms   "
          "total %7.3f s [%7dx]\n",
          rec.total / total_time * 100.0f, rec.name.c_str(), rec.min,
          rec.total / rec.counter, rec.max, rec.total / 1000.0f, rec.counter);
    }
  }
};

#if defined(TLANG_GPU)

#include <cuda_runtime.h>

class GPUProfiler : public ProfilerBase {
 public:
  cudaEvent_t current_stop;

  std::map<std::string, std::vector<std::pair<cudaEvent_t, cudaEvent_t>>>
      outstanding_events;

  void start(const std::string &kernel_name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    outstanding_events[kernel_name].push_back(std::make_pair(start, stop));
    current_stop = stop;
  }

  void stop() {
    cudaEventRecord(current_stop);
  }

  std::string title() override {
    return "GPU Profiler";
  }

  void sync() override {
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
  }

  static GPUProfiler &get_instance() {
    static GPUProfiler profiler;
    return profiler;
  }
};
#endif

class CPUProfiler : public ProfilerBase {
 public:
  double start_t;
  std::string event_name;

  void sync() override {
  }

  std::string title() override {
    return "CPU Profiler";
  }

  void start(const std::string &kernel_name) {
    start_t = get_time();
    event_name = kernel_name;
  }

  void stop() {
    auto t = get_time() - start_t;
    auto ms = t * 1000.0;
    auto it =
        std::find_if(records.begin(), records.end(),
                     [&](ProfileRecord &r) { return r.name == event_name; });
    if (it == records.end()) {
      records.emplace_back(event_name);
      it = std::prev(records.end());
    }
    it->insert_sample(ms);
    total_time += ms;
  }
};

TLANG_NAMESPACE_END
