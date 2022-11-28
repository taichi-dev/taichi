#include "taichi/rhi/vulkan/vulkan_stream.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_semaphore.h"

namespace taichi::lang {
namespace vulkan {

VulkanStream::VulkanStream(VulkanDevice &device,
                           VkQueue queue,
                           uint32_t queue_family_index)
    : device_(device), queue_(queue), queue_family_index_(queue_family_index) {
  command_pool_ = vkapi::create_command_pool(
      device_.vk_device(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      queue_family_index);
}

VulkanStream::~VulkanStream() {
}

std::unique_ptr<CommandList> VulkanStream::new_command_list() {
  vkapi::IVkCommandBuffer buffer =
      vkapi::allocate_command_buffer(command_pool_);

  return std::make_unique<VulkanCommandList>(&device_, this, buffer);
}

StreamSemaphore VulkanStream::submit(
    CommandList *cmdlist_,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  VulkanCommandList *cmdlist = static_cast<VulkanCommandList *>(cmdlist_);
  vkapi::IVkCommandBuffer buffer = cmdlist->finalize();
  vkapi::IVkQueryPool query_pool = cmdlist->vk_query_pool();

  /*
  if (in_flight_cmdlists_.find(buffer) != in_flight_cmdlists_.end()) {
    TI_ERROR("Can not submit command list that is still in-flight");
    return;
  }
  */

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &buffer->buffer;

  std::vector<VkSemaphore> vk_wait_semaphores;
  std::vector<VkPipelineStageFlags> vk_wait_stages;

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
    vk_wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    buffer->refs.push_back(sema->vkapi_ref);
  }

  submit_info.pWaitSemaphores = vk_wait_semaphores.data();
  submit_info.waitSemaphoreCount = vk_wait_semaphores.size();
  submit_info.pWaitDstStageMask = vk_wait_stages.data();

  auto semaphore = vkapi::create_semaphore(buffer->device, 0);
  buffer->refs.push_back(semaphore);

  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &semaphore->semaphore;

  auto fence = vkapi::create_fence(buffer->device, 0);

  // Resource tracking, check previously submitted commands
  // FIXME: Figure out why it doesn't work
  /*
  std::remove_if(submitted_cmdbuffers_.begin(), submitted_cmdbuffers_.end(),
                 [&](const TrackedCmdbuf &tracked) {
                   // If fence is signaled, cmdbuf has completed
                   VkResult res =
                       vkGetFenceStatus(buffer->device, tracked.fence->fence);
                   return res == VK_SUCCESS;
    });
  */

  submitted_cmdbuffers_.push_back(TrackedCmdbuf{fence, buffer, query_pool});

  BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(queue_, /*submitCount=*/1, &submit_info,
                                      /*fence=*/fence->fence),
                        "failed to submit command buffer");

  return std::make_shared<VulkanStreamSemaphoreObject>(semaphore);
}

StreamSemaphore VulkanStream::submit_synced(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  auto sema = submit(cmdlist, wait_semaphores);
  command_sync();
  return sema;
}

void VulkanStream::command_sync() {
  vkQueueWaitIdle(queue_);

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device_.vk_physical_device(), &props);

  for (const auto &cmdbuf : submitted_cmdbuffers_) {
    if (cmdbuf.query_pool == nullptr) {
      continue;
    }

    double duration_us = 0.0;

// Workaround for MacOS: https://github.com/taichi-dev/taichi/issues/5888
#if !defined(__APPLE__)
    uint64_t t[2];
    vkGetQueryPoolResults(device_.vk_device(), cmdbuf.query_pool->query_pool, 0,
                          2, sizeof(uint64_t) * 2, &t, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    duration_us = (t[1] - t[0]) * props.limits.timestampPeriod / 1000.0;
#endif

    device_time_elapsed_us_ += duration_us;
  }

  submitted_cmdbuffers_.clear();
}

double VulkanStream::device_time_elapsed_us() const {
  return device_time_elapsed_us_;
}

}  // namespace vulkan
}  // namespace taichi::lang
