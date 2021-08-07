#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>

#include "vk_mem_alloc.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkBufferWithMemory {
 public:
  VkBufferWithMemory(VmaAllocator &allocator,
                     size_t size,
                     VkBufferUsageFlags usage,
                     bool host_write = false,
                     bool host_read = false,
                     bool sparse = false);

  // Just use std::unique_ptr to save all the trouble from crafting move ctors
  // on our own
  VkBufferWithMemory(const VkBufferWithMemory &) = delete;
  VkBufferWithMemory &operator=(const VkBufferWithMemory &) = delete;
  VkBufferWithMemory(VkBufferWithMemory &&) = delete;
  VkBufferWithMemory &operator=(VkBufferWithMemory &&) = delete;

  ~VkBufferWithMemory();

  VkBuffer buffer() const {
    return buffer_;
  }

  VkDeviceSize size() const {
    return size_;
  }

  VkDeviceSize offset_in_mem() const {
    return alloc_info_.offset;
  }

  class Mapped {
   public:
    explicit Mapped(VkBufferWithMemory *buf) : buf_(buf), data_(nullptr) {
      vmaMapMemory(buf->allocator_, buf->allocation_, &data_);
    }

    ~Mapped() {
      vmaUnmapMemory(buf_->allocator_, buf_->allocation_);
    }

    void *data() const {
      return data_;
    }

   private:
    VkBufferWithMemory *const buf_;  // not owned
    void *data_;
  };

  Mapped map_mem() {
    return Mapped(this);
  }

 private:
  friend class Mapped;

  VmaAllocator& allocator_;
  VmaAllocation allocation_;
  VmaAllocationInfo alloc_info_;
  VkBuffer buffer_;
  VkDeviceSize size_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
