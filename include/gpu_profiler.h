#if defined(TLANG_GPU)

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.h"

TLANG_NAMESPACE_BEGIN

class GPUProfiler {
 public:
  cudaEvent_t current_stop;

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

  std::vector<ProfileRecord> records;

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

  void print() {
    sync();
    printf("GPU Profiler:\n");
    for (auto &rec : records) {
      printf("    %30s     min %7.3f ms   avg %7.3f ms    max %7.3f ms\n",
             rec.name.c_str(), rec.min, rec.total / rec.counter, rec.max);
    }
  }

  void sync() {
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
      }
    }
    outstanding_events.clear();
  }

  static GPUProfiler &get_instance() {
    static GPUProfiler profiler;
    return profiler;
  }
};

TLANG_NAMESPACE_END

#endif
