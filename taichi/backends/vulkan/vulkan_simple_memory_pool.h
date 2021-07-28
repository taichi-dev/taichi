#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>

namespace taichi {
namespace lang {
namespace vulkan {

class VkBufferWithMemory {
 public:
  VkBufferWithMemory(VkDevice device,
                     VkBuffer buffer,
                     VkDeviceMemory mem,
                     VkDeviceSize size,
                     VkDeviceSize offset);

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
    return offset_in_mem_;
  }

  class Mapped {
   public:
    explicit Mapped(VkBufferWithMemory *buf) : buf_(buf), data_(nullptr) {
      vkMapMemory(buf_->device_, buf_->backing_memory_, buf_->offset_in_mem(),
                  buf_->size(), /*flags=*/0, &data_);
    }

    ~Mapped() {
      vkUnmapMemory(buf_->device_, buf_->backing_memory_);
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

  VkDevice device_{VK_NULL_HANDLE};
  VkBuffer buffer_{VK_NULL_HANDLE};
  VkDeviceMemory backing_memory_{VK_NULL_HANDLE};
  VkDeviceSize size_{0};
  VkDeviceSize offset_in_mem_{0};
};

// TODO: Use
// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/custom_memory_pools.html
class LinearVkMemoryPool {
 public:
  struct Params {
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkMemoryPropertyFlags required_properties;
    VkDeviceSize pool_size{0};
    uint32_t compute_queue_family_index{0};
    VkBufferCreateInfo buffer_creation_template{};
  };

  LinearVkMemoryPool(const Params &params, VkDeviceMemory mem, uint32_t mti);

  ~LinearVkMemoryPool();

  static std::unique_ptr<LinearVkMemoryPool> try_make(Params params);

  std::unique_ptr<VkBufferWithMemory> alloc_and_bind(VkDeviceSize buf_size);

 private:
  static std::optional<uint32_t> find_memory_type(const Params &params);

  VkDevice device_{VK_NULL_HANDLE};  // not owned
  VkDeviceMemory memory_{VK_NULL_HANDLE};
  uint32_t memory_type_index_{0};
  uint32_t compute_queue_family_index_{0};
  VkBufferCreateInfo buffer_creation_template_{};
  VkDeviceSize pool_size_{0};
  VkDeviceSize next_{0};
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
