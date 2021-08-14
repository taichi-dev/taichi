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

void VulkanDevice::init_vulkan_structs(Params &params) {
  instance_ = params.instance;
  device_ = params.device;
  physical_device_ = params.physical_device;
  compute_queue_ = params.compute_queue;
  compute_pool_ = params.compute_pool;
  graphics_queue_ = params.graphics_queue;
  graphics_pool_ = params.graphics_pool;

  create_vma_allocator();
}

VulkanDevice::~VulkanDevice() {
  vmaDestroyAllocator(allocator_);
}

DeviceAllocation VulkanDevice::allocate_memory(const AllocParams& params) {
  DeviceAllocation handle;

  handle.device = this;
  handle.alloc_id = (alloc_cnt_++);

  allocations_[handle.alloc_id] = {};
  AllocationInternal& alloc = allocations_[handle.alloc_id];

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = params.size;
  // FIXME: How to express this in a backend-neutral way?
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

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

  vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &alloc.buffer, &alloc.allocation,
                  &alloc.alloc_info);

  TI_INFO("Allocated id {} size {}", handle.alloc_id, alloc.alloc_info.size);

  return handle;
}

void VulkanDevice::dealloc_memory(DeviceAllocation allocation) {
  AllocationInternal& alloc = allocations_[allocation.alloc_id];
  
  TI_INFO("Dealloc id {} size {}", allocation.alloc_id, alloc.alloc_info.size);

  vmaDestroyBuffer(allocator_, alloc.buffer, alloc.allocation);

  allocations_.erase(allocation.alloc_id);
}

void* VulkanDevice::map_range(DevicePtr ptr, uint64_t size) {
  AllocationInternal &alloc_int = allocations_.at(ptr.allocation->alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr, "Memory can not be mapped multiple times");

  vkMapMemory(device_, alloc_int.alloc_info.deviceMemory, alloc_int.alloc_info.offset + ptr.offset, size, 0, &alloc_int.mapped);

  return alloc_int.mapped;
}

void* VulkanDevice::map(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr, "Memory can not be mapped multiple times");

  vkMapMemory(device_, alloc_int.alloc_info.deviceMemory, alloc_int.alloc_info.offset, alloc_int.alloc_info.size, 0, &alloc_int.mapped);

  return alloc_int.mapped;  
}

void VulkanDevice::unmap(DevicePtr ptr) {
  AllocationInternal &alloc_int = allocations_.at(ptr.allocation->alloc_id);
  
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

void VulkanDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {

}

std::tuple<VkDeviceMemory, size_t, size_t> VulkanDevice::get_vkmemory_offset_size(const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  return std::make_tuple(alloc_int.alloc_info.deviceMemory, alloc_int.alloc_info.offset, alloc_int.alloc_info.size);
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
  vk_vma_functions.vkFlushMappedMemoryRanges =
      table.vkFlushMappedMemoryRanges;
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
      PFN_vkGetPhysicalDeviceMemoryProperties2KHR(
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2KHR"));

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = this->get_cap(
      DeviceCapability::vk_api_version);
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;
  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  vmaCreateAllocator(&allocatorInfo, &allocator_);
}

}
}
}