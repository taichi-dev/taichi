#pragma once

#include "taichi/system/timeline.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/vulkan/vulkan_api.h"

#include <deque>
#include <string>
#include <stdint.h>

namespace taichi::lang {

class EventToolkit;

class VulkanProfiler : public KernelProfilerBase {
 public:
  explicit VulkanProfiler(){}

  void clear() override {}

  void sync() override {}

  void update() override {
    size_t n_records = dispatched_kernels.size();
    for(int i = 0; i < n_records; ++i) {
      auto [kernel_name, query_pool] = dispatched_kernels.front();
      printf("pop %s\n", kernel_name.c_str());
      dispatched_kernels.pop_front();
    }
  }

  void record_dispatch(const std::string &kernel_name, const vkapi::IVkQueryPool query_pool) {
    printf("push %s\n", kernel_name.c_str());
    kernel_names_in_batch_.push_back(kernel_name);
    dispatched_kernels.push_back({kernel_name, query_pool});
  };

  void record_time(double duration_ms) {
    auto kernel_name = kernel_names_in_batch_.front();
    printf("pop %s\n", kernel_name.c_str());
    kernel_names_in_batch_.pop_front();

  }

  size_t batch_size() {
    return kernel_names_in_batch_.size();
  }

 private:
  // The Vulkan backend dispatches multiple kernel funcs into one single command list.
  // However, we cannot get the timestamps before commandlist is well synced.
  // We keep all the kernel names here in order to properly get the duration time for each kernel function.
  std::deque<std::string> kernel_names_in_batch_;
  std::deque<std::pair<std::string, vkapi::IVkQueryPool>> dispatched_kernels;
};


}  // namespace taichi::lang
