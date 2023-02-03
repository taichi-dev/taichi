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
      dispatched_kernels.pop_front();
      double duration_time = 1.0;

      uint64_t t[2];
      vkGetQueryPoolResults(vk_device_, query_pool->query_pool,
                            0, 2, sizeof(uint64_t) * 2, &t, sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      duration_time = (t[1] - t[0]) * vk_timestamp_period / 1000.0 / 1000.0;

      // Trace record
      KernelProfileTracedRecord record;
      record.name = kernel_name;
      record.kernel_elapsed_time_in_ms = duration_time;
      traced_records_.push_back(record);
      // Count record
      auto it =
          std::find_if(statistical_results_.begin(), statistical_results_.end(),
                       [&](KernelProfileStatisticalResult &r) {
                         return r.name == record.name;
                       });
      if (it == statistical_results_.end()) {
        statistical_results_.emplace_back(record.name);
        it = std::prev(statistical_results_.end());
      }
      it->insert_record(duration_time);
      total_time_ms_ += duration_time;
    }
  }

  void record_dispatch(const std::string &kernel_name, const vkapi::IVkQueryPool query_pool) {
    // printf("push %s\n", kernel_name.c_str());
    dispatched_kernels.push_back({kernel_name, query_pool});
  };

   void set_vk_device(const VkDevice &device) {
    vk_device_ = device;
    // VkPhysicalDeviceProperties props{};
    // vkGetPhysicalDeviceProperties(vk_device_->vk_physical_device(), &props);
    // vk_timestamp_period = props.limits.timestampPeriod;
   }

   void set_vk_timestamp_period(double timestamp_period) {
    vk_timestamp_period = timestamp_period;
   }
 private:
  // The Vulkan backend dispatches multiple kernel funcs into one single command list.
  // However, we cannot get the timestamps before commandlist is well synced.
  // We keep all the kernel names here in order to properly get the duration time for each kernel function.
   std::deque<std::pair<std::string, vkapi::IVkQueryPool>> dispatched_kernels;
   VkDevice vk_device_;
   int vk_timestamp_period;
};


}  // namespace taichi::lang
