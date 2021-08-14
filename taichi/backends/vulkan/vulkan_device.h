#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>

#include <taichi/backends/device.h>

#include "vk_mem_alloc.h"

namespace taichi {
namespace lang {
namespace vulkan
{

class VulkanDevice : public Device {
 public:
  struct Params {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool compute_pool;
    VkQueue graphics_queue;
    VkCommandPool graphics_pool;
  };

  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  DeviceAllocation allocate_memory(const AllocParams& params) override;
  void dealloc_memory(DeviceAllocation allocation) override;

  // Mapping can fail and will return nullptr
  void* map_range(DevicePtr ptr, uint64_t size) override;
  void* map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;
  
  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  // Vulkan specific functions
  VkDevice vk_device() const {
    return device_;
  }

  VkQueue graphics_queue() const {
    return graphics_queue_;
  }

  VkQueue compute_queue() const {
    return compute_queue_;
  }

  VkCommandPool graphics_cmd_pool() const {
    return graphics_pool_;
  }

  VkCommandPool compute_cmd_pool() const {
    return compute_pool_;
  }

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(const DeviceAllocation &alloc) const;

  VkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

 private:
  void create_vma_allocator();

  VkInstance instance_;
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VmaAllocator allocator_;

  VkQueue compute_queue_;
  VkCommandPool compute_pool_;

  VkQueue graphics_queue_;
  VkCommandPool graphics_pool_;

  struct AllocationInternal {
    VmaAllocation allocation;
    VmaAllocationInfo alloc_info;
    VkBuffer buffer;
    void* mapped{nullptr};
  };
  
  std::unordered_map<uint32_t, AllocationInternal> allocations_;

  uint32_t alloc_cnt_ = 0;
};

}
}
}