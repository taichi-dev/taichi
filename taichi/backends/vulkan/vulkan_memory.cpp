#include "taichi/backends/vulkan/vulkan_memory.h"

#include "taichi/math/arithmetic.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/common/logging.h"

#include "vk_mem_alloc.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace {

static constexpr VkDeviceSize kAlignment = 256;

VkDeviceSize roundup_aligned(VkDeviceSize size) {
  return iroundup(size, kAlignment);
}

}  // namespace

VkBufferWithMemory::VkBufferWithMemory(VmaAllocator &allocator,
                                       size_t size,
                                       VkBufferUsageFlags usage,
                                       bool host_write,
                                       bool host_read,
                                       bool sparse)
    : allocator_(allocator), size_(size) {
  TI_ASSERT(size_ > 0);

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = size;
  buffer_info.usage = usage;

  VmaAllocationCreateInfo alloc_info{};

  if (host_read && host_write) {
    // This should be the unified memory on integrated GPUs
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  } else if (host_read) {
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
  } else if (host_write) {
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
  } else {
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  }

  vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer_, &allocation_,
                  &alloc_info_);
}

VkBufferWithMemory::~VkBufferWithMemory() {
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
