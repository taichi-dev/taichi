#include "taichi/backends/vulkan/vulkan_simple_memory_pool.h"

#include "taichi/math/arithmetic.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace {

static constexpr VkDeviceSize kAlignment = 256;

VkDeviceSize roundup_aligned(VkDeviceSize size) {
  return iroundup(size, kAlignment);
}

}  // namespace

VkBufferWithMemory::VkBufferWithMemory(VkDevice device,
                                       VkBuffer buffer,
                                       VkDeviceMemory mem,
                                       VkDeviceSize size,
                                       VkDeviceSize offset)
    : device_(device),
      buffer_(buffer),
      backing_memory_(mem),
      size_(size),
      offset_in_mem_(offset) {
  TI_ASSERT(buffer_ != VK_NULL_HANDLE);
  TI_ASSERT(size_ > 0);
  TI_ASSERT(backing_memory_ != VK_NULL_HANDLE);
}

VkBufferWithMemory::~VkBufferWithMemory() {
  if (buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, buffer_, kNoVkAllocCallbacks);
  }
}

LinearVkMemoryPool::LinearVkMemoryPool(const Params &params,
                                       VkDeviceMemory mem,
                                       uint32_t mti)
    : device_(params.device),
      memory_(mem),
      memory_type_index_(mti),
      compute_queue_family_index_(params.compute_queue_family_index),
      buffer_creation_template_(params.buffer_creation_template),
      pool_size_(params.pool_size),
      next_(0) {
  buffer_creation_template_.size = 0;
  buffer_creation_template_.queueFamilyIndexCount = 1;
  buffer_creation_template_.pQueueFamilyIndices = &compute_queue_family_index_;
}

LinearVkMemoryPool::~LinearVkMemoryPool() {
  if (memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, memory_, kNoVkAllocCallbacks);
  }
}

// static
std::unique_ptr<LinearVkMemoryPool> LinearVkMemoryPool::try_make(
    Params params) {
  params.pool_size = roundup_aligned(params.pool_size);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = params.pool_size;
  const auto mem_type_index = find_memory_type(params);
  if (!mem_type_index.has_value()) {
    return nullptr;
  }
  alloc_info.memoryTypeIndex = mem_type_index.value();
  VkDeviceMemory mem;
  if (vkAllocateMemory(params.device, &alloc_info, kNoVkAllocCallbacks, &mem) !=
      VK_SUCCESS) {
    return nullptr;
  }
  return std::make_unique<LinearVkMemoryPool>(params, mem,
                                              alloc_info.memoryTypeIndex);
}

std::unique_ptr<VkBufferWithMemory> LinearVkMemoryPool::alloc_and_bind(
    VkDeviceSize buf_size) {
  buf_size = roundup_aligned(buf_size);
  if (pool_size_ <= (next_ + buf_size)) {
    TI_WARN("Vulkan memory pool exhausted, max size={}", pool_size_);
    return nullptr;
  }

  VkBuffer buffer;
  buffer_creation_template_.size = buf_size;
  BAIL_ON_VK_BAD_RESULT(vkCreateBuffer(device_, &buffer_creation_template_,
                                       kNoVkAllocCallbacks, &buffer),
                        "failed to create buffer");
  buffer_creation_template_.size = 0;  // reset
  const auto offset_in_mem = next_;
  next_ += buf_size;
  BAIL_ON_VK_BAD_RESULT(
      vkBindBufferMemory(device_, buffer, memory_, offset_in_mem),
      "failed to bind buffer to memory");

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);
  TI_ASSERT(mem_requirements.memoryTypeBits & (1 << memory_type_index_));
  TI_ASSERT_INFO((buf_size % mem_requirements.alignment) == 0,
                 "buf_size={} required alignment={}", buf_size,
                 mem_requirements.alignment);
  return std::make_unique<VkBufferWithMemory>(device_, buffer, memory_,
                                              buf_size, offset_in_mem);
}

// static
std::optional<uint32_t> LinearVkMemoryPool::find_memory_type(
    const Params &params) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(params.physical_device, &mem_properties);
  auto satisfies = [&](int i) -> bool {
    const auto &mem_type = mem_properties.memoryTypes[i];
    if ((mem_type.propertyFlags & params.required_properties) !=
        params.required_properties) {
      return false;
    }
    if (mem_properties.memoryHeaps[mem_type.heapIndex].size <=
        params.pool_size) {
      return false;
    }
    return true;
  };

  for (int i = 0; i < mem_properties.memoryTypeCount; ++i) {
    if (satisfies(i)) {
      return i;
    }
  }
  return std::nullopt;
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
