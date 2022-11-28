#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang {
namespace vulkan {

class VulkanResourceBinder : public ResourceBinder {
 public:
  struct Binding {
    VkDescriptorType type;
    DevicePtr ptr;
    VkDeviceSize size;
    union {
      VkSampler sampler{VK_NULL_HANDLE};  // used only for images
      int image_lod;
    };

    bool operator==(const Binding &other) const {
      return other.type == type && other.ptr == ptr && other.size == size &&
             other.sampler == sampler;
    }

    bool operator!=(const Binding &other) const {
      return !(*this == other);
    }
  };

  struct Set {
    std::unordered_map<uint32_t, Binding> bindings;

    // The compare function is for the hashmap to locate a set layout
    bool operator==(const Set &other) const {
      if (other.bindings.size() != bindings.size()) {
        return false;
      }
      for (auto &pair : bindings) {
        auto other_binding_iter = other.bindings.find(pair.first);
        if (other_binding_iter == other.bindings.end()) {
          return false;
        }
        const Binding &other_binding = other_binding_iter->second;
        if (other_binding.type != pair.second.type) {
          return false;
        }
      }
      return true;
    }

    bool operator!=(const Set &other) const {
      return !(*this == other);
    }
  };

  struct SetLayoutHasher {
    std::size_t operator()(const Set &set) const {
      // TODO: Come up with a better hash
      size_t hash = 0;
      for (const auto &pair : set.bindings) {
        hash = (hash ^ size_t(pair.second.type)) ^ size_t(pair.first);
      }
      return hash;
    }
  };

  struct DescSetCmp {
    bool operator()(const Set &a, const Set &b) const {
      if (a.bindings.size() != b.bindings.size()) {
        return false;
      }
      for (auto &pair : a.bindings) {
        auto other_binding_iter = b.bindings.find(pair.first);
        if (other_binding_iter == b.bindings.end()) {
          return false;
        }
        const Binding &other_binding = other_binding_iter->second;
        if (other_binding != pair.second) {
          return false;
        }
      }
      return true;
    }
  };

  struct DescSetHasher {
    std::size_t operator()(const Set &set) const {
      // TODO: Come up with a better hash
      size_t hash = 0;
      for (const auto &pair : set.bindings) {
        size_t binding_hash = 0;
        uint32_t *u32_ptr = (uint32_t *)&pair.second;
        static_assert(
            sizeof(VulkanResourceBinder::Binding) % sizeof(uint32_t) == 0,
            "sizeof(VulkanResourceBinder::Binding) is not a multiple of 4");
        size_t n = sizeof(VulkanResourceBinder::Binding) / sizeof(uint32_t);
        for (size_t i = 0; i < n; i++) {
          binding_hash = binding_hash ^ u32_ptr[i];
          binding_hash = (binding_hash << 7) | (binding_hash >> (64 - 7));
        }
        binding_hash = binding_hash ^ pair.first;
        binding_hash =
            (binding_hash << pair.first) | (binding_hash >> (64 - pair.first));
        hash = hash ^ binding_hash;
      }
      return hash;
    }
  };

  struct VulkanBindings : public Bindings {
    std::vector<
        std::pair<vkapi::IVkDescriptorSetLayout, vkapi::IVkDescriptorSet>>
        sets;
  };

  explicit VulkanResourceBinder(
      VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_COMPUTE);
  ~VulkanResourceBinder() override;

  std::unique_ptr<Bindings> materialize() override;

  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override;
  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override;
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override;
  void image(uint32_t set,
             uint32_t binding,
             DeviceAllocation alloc,
             ImageSamplerConfig sampler_config) override;
  void rw_image(uint32_t set,
                uint32_t binding,
                DeviceAllocation alloc,
                int lod) override;
  void vertex_buffer(DevicePtr ptr, uint32_t binding = 0) override;
  void index_buffer(DevicePtr ptr, size_t index_width) override;

  void write_to_set(uint32_t index,
                    VulkanDevice &device,
                    vkapi::IVkDescriptorSet set);
  Set &get_set(uint32_t index) {
    return sets_[index];
  }
  std::unordered_map<uint32_t, Set> &get_sets() {
    return sets_;
  }
  std::unordered_map<uint32_t, DevicePtr> &get_vertex_buffers() {
    return vertex_buffers_;
  }
  std::pair<DevicePtr, VkIndexType> get_index_buffer() {
    return std::make_pair(index_buffer_, index_type_);
  }

  void lock_layout();

 private:
  std::unordered_map<uint32_t, Set> sets_;
  bool layout_locked_{false};
  VkPipelineBindPoint bind_point_;

  std::unordered_map<uint32_t, DevicePtr> vertex_buffers_;
  DevicePtr index_buffer_{kDeviceNullPtr};
  VkIndexType index_type_;
};

} // namespace vulkan
} // namespace taichi::lang
