#include "taichi/backends/vulkan/vulkan_api.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/loader.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
namespace vulkan {

VulkanPipeline::VulkanPipeline(const Params &params)
    : device_(params.device->vk_device()), name_(params.name) {
  create_descriptor_set_layout(params);
  create_compute_pipeline(params);
  create_descriptor_pool(params);
  create_descriptor_sets(params);
}

VulkanPipeline::~VulkanPipeline() {
  vkDestroyDescriptorPool(device_, descriptor_pool_, kNoVkAllocCallbacks);
  vkDestroyPipeline(device_, pipeline_, kNoVkAllocCallbacks);
  vkDestroyPipelineLayout(device_, pipeline_layout_, kNoVkAllocCallbacks);
  vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_,
                               kNoVkAllocCallbacks);
}

VkShaderModule VulkanPipeline::create_shader_module(VkDevice device,
                                                    const SpirvCodeView &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size;
  create_info.pCode = code.data;

  VkShaderModule shader_module;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateShaderModule(device, &create_info, kNoVkAllocCallbacks,
                           &shader_module),
      "failed to create shader module");
  return shader_module;
}

void VulkanPipeline::create_descriptor_set_layout(const Params &params) {
  const auto &buffer_binds = params.buffer_bindings;
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
  layout_bindings.reserve(buffer_binds.size());
  for (const auto &bb : buffer_binds) {
    VkDescriptorSetLayoutBinding layout_binding{};
    layout_binding.binding = bb.binding;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layout_binding.pImmutableSamplers = nullptr;
    layout_bindings.push_back(layout_binding);
  }

  VkDescriptorSetLayoutCreateInfo layout_create_info{};
  layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_create_info.bindingCount = layout_bindings.size();
  layout_create_info.pBindings = layout_bindings.data();

  BAIL_ON_VK_BAD_RESULT(
      vkCreateDescriptorSetLayout(device_, &layout_create_info,
                                  kNoVkAllocCallbacks, &descriptor_set_layout_),
      "failed to create descriptor set layout");
}

void VulkanPipeline::create_compute_pipeline(const Params &params) {
  VkShaderModule shader_module = create_shader_module(device_, params.code);

  VkPipelineShaderStageCreateInfo shader_stage_info{};
  shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage_info.module = shader_module;
#pragma message("Shader storage info: pName is hardcoded to \"main\"")
  shader_stage_info.pName = "main";

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;
  BAIL_ON_VK_BAD_RESULT(
      vkCreatePipelineLayout(device_, &pipeline_layout_info,
                             kNoVkAllocCallbacks, &pipeline_layout_),
      "failed to create pipeline layout");

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = shader_stage_info;
  pipeline_info.layout = pipeline_layout_;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateComputePipelines(device_, /*pipelineCache=*/VK_NULL_HANDLE,
                               /*createInfoCount=*/1, &pipeline_info,
                               kNoVkAllocCallbacks, &pipeline_),
      "failed to create pipeline");

  vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
}

void VulkanPipeline::create_descriptor_pool(const Params &params) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  // This is the total number of descriptors we will allocate from this pool,
  // across all the descriptor sets.
  // https://stackoverflow.com/a/51716660/12003165
  pool_size.descriptorCount = params.buffer_bindings.size();

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateDescriptorPool(device_, &pool_info, kNoVkAllocCallbacks,
                             &descriptor_pool_),
      "failed to create descriptor pool");
}

void VulkanPipeline::create_descriptor_sets(const Params &params) {
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout_;

  BAIL_ON_VK_BAD_RESULT(
      vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_),
      "failed to allocate descriptor set");

  const auto &buffer_binds = params.buffer_bindings;
  std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.reserve(buffer_binds.size());
  for (const auto &bb : buffer_binds) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = bb.buffer;
    // Note that this is the offset within the buffer itself, not the offset
    // of this buffer within its backing memory!
    buffer_info.offset = 0;
    // https://github.com/apache/tvm/blob/d288bbc5df3660355adbf97f2f84ecd232e269ff/src/runtime/vulkan/vulkan.cc#L1073
    buffer_info.range = VK_WHOLE_SIZE;
    descriptor_buffer_infos.push_back(buffer_info);
  }

  std::vector<VkWriteDescriptorSet> descriptor_writes;
  descriptor_writes.reserve(descriptor_buffer_infos.size());
  for (int i = 0; i < buffer_binds.size(); ++i) {
    const auto &bb = buffer_binds[i];

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = bb.binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor_buffer_infos[i];
    write.pImageInfo = nullptr;
    write.pTexelBufferView = nullptr;
    descriptor_writes.push_back(write);
  }

  vkUpdateDescriptorSets(device_,
                         /*descriptorWriteCount=*/descriptor_writes.size(),
                         descriptor_writes.data(), /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
}

VulkanCommandList::VulkanCommandList(VulkanDevice *ti_device,
                                     VkDevice device,
                                     VkCommandBuffer buffer)
    : ti_device_(ti_device), device_(device), buffer_(buffer) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  vkBeginCommandBuffer(buffer, &info);
}

VulkanCommandList::~VulkanCommandList() {
  ti_device_->dealloc_command_list(this);
}

void VulkanCommandList::bind_pipeline(Pipeline *p) {
  auto pipeline = static_cast<VulkanPipeline *>(p);

  vkCmdBindPipeline(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipeline->pipeline());
  vkCmdBindDescriptorSets(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline->pipeline_layout(), 0, 1,
                          &pipeline->descriptor_set(), 0, nullptr);
}

void VulkanCommandList::bind_resources(ResourceBinder &binder) {
}

void VulkanCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  TI_ASSERT(ptr.device == ti_device_);

  VkBufferMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.buffer = ti_device_->get_vkbuffer(ptr);
  barrier.offset = ptr.offset;
  barrier.size = size;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/0, nullptr,
      /*bufferMemoryBarrierCount=*/1,
      /*pBufferMemoryBarriers=*/&barrier,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
}

void VulkanCommandList::buffer_barrier(DeviceAllocation alloc) {
  buffer_barrier(DevicePtr{alloc, 0}, VK_WHOLE_SIZE);
}

void VulkanCommandList::memory_barrier() {
  VkMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/1, &barrier,
      /*bufferMemoryBarrierCount=*/0,
      /*pBufferMemoryBarriers=*/nullptr,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
}

void VulkanCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  VkBufferCopy copy_region{};
  copy_region.srcOffset = src.offset;
  copy_region.dstOffset = dst.offset;
  copy_region.size = size;
  vkCmdCopyBuffer(buffer_, ti_device_->get_vkbuffer(src),
                  ti_device_->get_vkbuffer(dst), /*regionCount=*/1,
                  &copy_region);
}

void VulkanCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  vkCmdFillBuffer(buffer_, ti_device_->get_vkbuffer(ptr), ptr.offset, size,
                  data);
}

void VulkanCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  vkCmdDispatch(buffer_, x, y, z);
}

void VulkanCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  vkCmdDraw(buffer_, num_verticies, /*instanceCount=*/1, start_vertex,
            /*firstInstance=*/0);
}

void VulkanCommandList::draw_indexed(uint32_t num_indicies,
                                     uint32_t start_vertex,
                                     uint32_t start_index) {
  vkCmdDrawIndexed(buffer_, num_indicies, /*instanceCount=*/1, start_index,
                   start_vertex,
                   /*firstInstance=*/0);
}

VkCommandBuffer VulkanCommandList::finalize() {
  if (!finalized_) {
    vkEndCommandBuffer(buffer_);
    finalized_ = true;
  }
  return buffer_;
}

void VulkanDevice::init_vulkan_structs(Params &params) {
  instance_ = params.instance;
  device_ = params.device;
  physical_device_ = params.physical_device;
  compute_queue_ = params.compute_queue;
  compute_pool_ = params.compute_pool;
  graphics_queue_ = params.graphics_queue;
  graphics_pool_ = params.graphics_pool;

  create_vma_allocator();

  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
  BAIL_ON_VK_BAD_RESULT(vkCreateFence(device_, &fence_info, kNoVkAllocCallbacks,
                                      &cmd_sync_fence_),
                        "failed to create fence");
}

VulkanDevice::~VulkanDevice() {
  command_sync();

  vmaDestroyAllocator(allocator_);
  vkDestroyFence(device_, cmd_sync_fence_, kNoVkAllocCallbacks);
}

DeviceAllocation VulkanDevice::allocate_memory(const AllocParams &params) {
  DeviceAllocation handle;

  handle.device = this;
  handle.alloc_id = (alloc_cnt_++);

  allocations_[handle.alloc_id] = {};
  AllocationInternal &alloc = allocations_[handle.alloc_id];

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = params.size;
  // FIXME: How to express this in a backend-neutral way?
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo alloc_info{};

  if (params.host_read && params.host_write) {
    // This should be the unified memory on integrated GPUs
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  } else if (params.host_read) {
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
  } else if (params.host_write) {
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
  } else {
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  }

  vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &alloc.buffer,
                  &alloc.allocation, &alloc.alloc_info);

  return handle;
}

void VulkanDevice::dealloc_memory(DeviceAllocation allocation) {
  AllocationInternal &alloc = allocations_[allocation.alloc_id];

  vmaDestroyBuffer(allocator_, alloc.buffer, alloc.allocation);

  allocations_.erase(allocation.alloc_id);
}

void *VulkanDevice::map_range(DevicePtr ptr, uint64_t size) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
              alloc_int.alloc_info.offset + ptr.offset, size, 0,
              &alloc_int.mapped);

  return alloc_int.mapped;
}

void *VulkanDevice::map(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
              alloc_int.alloc_info.offset, alloc_int.alloc_info.size, 0,
              &alloc_int.mapped);

  return alloc_int.mapped;
}

void VulkanDevice::unmap(DevicePtr ptr) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  alloc_int.mapped = nullptr;
}

void VulkanDevice::unmap(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  alloc_int.mapped = nullptr;
}

void VulkanDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
}

std::unique_ptr<CommandList> VulkanDevice::new_command_list() {
  VkCommandBuffer buffer = VK_NULL_HANDLE;

  if (free_cmdbuffers_.size()) {
    buffer = free_cmdbuffers_.back();
    free_cmdbuffers_.pop_back();
  } else {
    // FIXME: Proper multi queue support, instead of defaulting to compute
    // command pool
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = compute_cmd_pool();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    BAIL_ON_VK_BAD_RESULT(
        vkAllocateCommandBuffers(device_, &alloc_info, &buffer),
        "failed to allocate command buffer");
  }

  return std::make_unique<VulkanCommandList>(this, device_, buffer);
}

void VulkanDevice::dealloc_command_list(CommandList *cmdlist) {
  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmdlist)->finalize();
  if (in_flight_cmdlists_.find(buffer) == in_flight_cmdlists_.end()) {
    // Not in flight
    free_cmdbuffers_.push_back(buffer);
  } else {
    // In flight
    dealloc_cmdlists_.push_back(buffer);
  }
}

void VulkanDevice::submit(CommandList *cmdlist) {
  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmdlist)->finalize();

  /*
  if (in_flight_cmdlists_.find(buffer) != in_flight_cmdlists_.end()) {
    TI_ERROR("Can not submit command list that is still in-flight");
    return;
  }
  */

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &buffer;

  // FIXME: Reuse fences as well?
  VkFence fence;
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
  BAIL_ON_VK_BAD_RESULT(
      vkCreateFence(device_, &fence_info, kNoVkAllocCallbacks, &fence),
      "failed to create fence");

  in_flight_cmdlists_.insert({buffer, fence});

  BAIL_ON_VK_BAD_RESULT(
      vkQueueSubmit(compute_queue(), /*submitCount=*/1, &submit_info,
                    /*fence=*/fence),
      "failed to submit command buffer");
}

void VulkanDevice::submit_synced(CommandList *cmdlist) {
  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmdlist)->finalize();

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &buffer;

  BAIL_ON_VK_BAD_RESULT(
      vkQueueSubmit(compute_queue(), /*submitCount=*/1, &submit_info,
                    /*fence=*/cmd_sync_fence_),
      "failed to submit command buffer");

  // Timeout is in nanoseconds, 60s = 60,000ms = 60,000,000ns
  vkWaitForFences(device_, 1, &cmd_sync_fence_, true, (60 * 1000 * 1000));
}

void VulkanDevice::command_sync() {
  if (!in_flight_cmdlists_.size()) {
    return;
  }

  std::vector<VkFence> fences;
  fences.reserve(in_flight_cmdlists_.size());

  for (auto &pair : in_flight_cmdlists_) {
    fences.push_back(pair.second);
  }

  vkWaitForFences(device_, fences.size(), fences.data(), true,
                  (60 * 1000 * 1000));

  for (auto &pair : in_flight_cmdlists_) {
    vkDestroyFence(device_, pair.second, kNoVkAllocCallbacks);
  }

  in_flight_cmdlists_.clear();

  for (auto buf : dealloc_cmdlists_) {
    free_cmdbuffers_.push_back(buf);
  }
}

std::tuple<VkDeviceMemory, size_t, size_t>
VulkanDevice::get_vkmemory_offset_size(const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  return std::make_tuple(alloc_int.alloc_info.deviceMemory,
                         alloc_int.alloc_info.offset,
                         alloc_int.alloc_info.size);
}

VkBuffer VulkanDevice::get_vkbuffer(const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  return alloc_int.buffer;
}

void VulkanDevice::create_vma_allocator() {
  VolkDeviceTable table;
  VmaVulkanFunctions vk_vma_functions;

  volkLoadDeviceTable(&table, device_);
  vk_vma_functions.vkGetPhysicalDeviceProperties =
      PFN_vkGetPhysicalDeviceProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceProperties"));
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties =
      PFN_vkGetPhysicalDeviceMemoryProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties"));
  vk_vma_functions.vkAllocateMemory = table.vkAllocateMemory;
  vk_vma_functions.vkFreeMemory = table.vkFreeMemory;
  vk_vma_functions.vkMapMemory = table.vkMapMemory;
  vk_vma_functions.vkUnmapMemory = table.vkUnmapMemory;
  vk_vma_functions.vkFlushMappedMemoryRanges = table.vkFlushMappedMemoryRanges;
  vk_vma_functions.vkInvalidateMappedMemoryRanges =
      table.vkInvalidateMappedMemoryRanges;
  vk_vma_functions.vkBindBufferMemory = table.vkBindBufferMemory;
  vk_vma_functions.vkBindImageMemory = table.vkBindImageMemory;
  vk_vma_functions.vkGetBufferMemoryRequirements =
      table.vkGetBufferMemoryRequirements;
  vk_vma_functions.vkGetImageMemoryRequirements =
      table.vkGetImageMemoryRequirements;
  vk_vma_functions.vkCreateBuffer = table.vkCreateBuffer;
  vk_vma_functions.vkDestroyBuffer = table.vkDestroyBuffer;
  vk_vma_functions.vkCreateImage = table.vkCreateImage;
  vk_vma_functions.vkDestroyImage = table.vkDestroyImage;
  vk_vma_functions.vkCmdCopyBuffer = table.vkCmdCopyBuffer;
  vk_vma_functions.vkGetBufferMemoryRequirements2KHR =
      table.vkGetBufferMemoryRequirements2KHR;
  vk_vma_functions.vkGetImageMemoryRequirements2KHR =
      table.vkGetImageMemoryRequirements2KHR;
  vk_vma_functions.vkBindBufferMemory2KHR = table.vkBindBufferMemory2KHR;
  vk_vma_functions.vkBindImageMemory2KHR = table.vkBindImageMemory2KHR;
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties2KHR =
      PFN_vkGetPhysicalDeviceMemoryProperties2KHR(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties2KHR"));

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion =
      this->get_cap(DeviceCapability::vk_api_version);
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;
  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  vmaCreateAllocator(&allocatorInfo, &allocator_);
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi