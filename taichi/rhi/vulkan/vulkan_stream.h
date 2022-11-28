#pragma once
#include "taichi/rhi/device.h"  // TODO: (penguinliong) Split this up.
#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang {
namespace vulkan {

class VulkanStream : public Stream {
 public:
  VulkanStream(VulkanDevice &device,
               VkQueue queue,
               uint32_t queue_family_index);
  ~VulkanStream() override;

  std::unique_ptr<CommandList> new_command_list() override;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void command_sync() override;

  double device_time_elapsed_us() const override;

 private:
  struct TrackedCmdbuf {
    vkapi::IVkFence fence;
    vkapi::IVkCommandBuffer buf;
    vkapi::IVkQueryPool query_pool;
  };

  VulkanDevice &device_;
  VkQueue queue_;
  uint32_t queue_family_index_;

  // Command pools are per-thread
  vkapi::IVkCommandPool command_pool_;
  std::vector<TrackedCmdbuf> submitted_cmdbuffers_;
  double device_time_elapsed_us_;
};

}  // namespace vulkan
}  // namespace taichi::lang
