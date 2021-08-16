#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include "vk_mem_alloc.h"

#include <memory>
#include <optional>

#include <taichi/backends/device.h>

namespace taichi {
namespace lang {
namespace vulkan {

class VulkanDevice;
class VulkanResourceBinder;

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

class VulkanResourceBinder : public ResourceBinder {
 public:
  struct Binding {
    VkDescriptorType type;
    DevicePtr ptr;
    size_t size;
  };

  struct Set {
    std::unordered_map<uint32_t, Binding> bindings;

    // The compare function is for the hashmap to locate a set layout
    bool operator==(const Set &other) const {
      if (other.bindings.size() != bindings.size()) {
        return false;
      }
      for (auto &pair : bindings) {
        const Binding &other_binding = other.bindings.at(pair.first);
        if (other_binding.type != pair.second.type) {
          return false;
        }
      }
      return true;
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

  ~VulkanResourceBinder();
  void rw_buffer(uint32_t set, uint32_t binding, DevicePtr ptr, size_t size);
  void rw_buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc);
  void buffer(uint32_t set, uint32_t binding, DevicePtr ptr, size_t size);
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc);
  void vertex_buffer(DevicePtr ptr, uint32_t binding = 0);
  void index_buffer(DevicePtr ptr, size_t index_width);
  void framebuffer_color(DeviceAllocation image, uint32_t binding);
  void framebuffer_depth_stencil(DeviceAllocation image);

  void write_to_set(uint32_t index, VulkanDevice &device, VkDescriptorSet set);
  Set &get_set(uint32_t index) {
    return sets_[index];
  }
  std::unordered_map<uint32_t, Set> &get_sets() {
    return sets_;
  }

  void lock_layout();

 private:
  std::unordered_map<uint32_t, Set> sets_;
  bool layout_locked_{false};

  // TODO: Graphics pipeline stuff
};

// VulkanPipeline maps to a VkPipeline, or a SPIR-V module (a GLSL compute
// shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    SpirvCodeView code;
    std::string name{"Pipeline"};
  };

  explicit VulkanPipeline(const Params &params);
  ~VulkanPipeline();

  ResourceBinder *resource_binder() override {
    return &resource_binder_;
  }

  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }
  VkPipeline pipeline() const {
    return pipeline_;
  }
  const std::string &name() const {
    return name_;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_compute_pipeline(const Params &params);

  static VkShaderModule create_shader_module(VkDevice device,
                                             const SpirvCodeView &code);

  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  VulkanResourceBinder resource_binder_;
  std::vector<VkDescriptorSetLayout> set_layouts_;
  VkPipeline pipeline_{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VkDevice device,
                    VkCommandBuffer buffer);
  ~VulkanCommandList();

  void bind_pipeline(Pipeline *p) override;
  void bind_resources(ResourceBinder *binder) override;
  void buffer_barrier(DevicePtr ptr, size_t size) override;
  void buffer_barrier(DeviceAllocation alloc) override;
  void memory_barrier() override;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override;
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void draw_indexed(uint32_t num_indicies,
                    uint32_t start_vertex = 0,
                    uint32_t start_index = 0) override;

  // Vulkan specific functions
  VkCommandBuffer finalize();

 private:
  bool finalized_{false};
  VulkanDevice *ti_device_;
  VkDevice device_;
  VkCommandBuffer buffer_;
  VulkanPipeline *current_pipeline_{nullptr};

  std::vector<std::pair<VkDescriptorSetLayout, VkDescriptorSet>> desc_sets_;
};

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

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation allocation) override;

  // Mapping can fail and will return nullptr
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  std::unique_ptr<CommandList> new_command_list() override;
  void dealloc_command_list(CommandList *cmdlist) override;
  void submit(CommandList *cmdlist) override;
  void submit_synced(CommandList *cmdlist) override;

  void command_sync() override;

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

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(
      const DeviceAllocation &alloc) const;

  VkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

  VkDescriptorSetLayout get_desc_set_layout(VulkanResourceBinder::Set &set);
  VkDescriptorSet alloc_desc_set(VkDescriptorSetLayout layout);
  void dealloc_desc_set(VkDescriptorSetLayout layout, VkDescriptorSet set);

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

  // Memory allocation
  struct AllocationInternal {
    VmaAllocation allocation;
    VmaAllocationInfo alloc_info;
    VkBuffer buffer;
    void *mapped{nullptr};
  };

  std::unordered_map<uint32_t, AllocationInternal> allocations_;

  uint32_t alloc_cnt_ = 0;

  // Command buffer tracking & allocation
  VkFence cmd_sync_fence_;

  std::unordered_multimap<VkCommandBuffer, VkFence> in_flight_cmdlists_;
  std::vector<VkCommandBuffer> dealloc_cmdlists_;
  std::vector<VkCommandBuffer> free_cmdbuffers_;

  // Descriptors / Layouts / Pools
  struct DescPool {
    VkDescriptorPool pool;
    std::vector<VkDescriptorSet> free_sets;
  };

  std::unordered_map<VulkanResourceBinder::Set,
                     VkDescriptorSetLayout,
                     VulkanResourceBinder::SetLayoutHasher>
      desc_set_layouts_;

  std::unordered_map<VkDescriptorSetLayout, DescPool> desc_set_pools_;

  std::unordered_multimap<VkDescriptorSet, VkFence> in_flight_desc_sets_;
  std::vector<std::pair<DescPool *, VkDescriptorSet>> dealloc_desc_sets_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
