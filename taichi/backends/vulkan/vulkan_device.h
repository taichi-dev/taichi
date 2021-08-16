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
namespace vulkan {

class VulkanDevice;

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

// VulkanPipeline maps to a VkPipeline, or a SPIR-V module (a GLSL compute
// shader). Because Taichi's buffers are all pre-allocated upon startup, we
// only need to set up the descriptor set (i.e., bind the buffers via
// VkWriteDescriptorSet) once during the pipeline initialization.
class VulkanPipeline : public Pipeline {
 public:
  struct BufferBinding {
    VkBuffer buffer{VK_NULL_HANDLE};
    uint32_t binding{0};
  };

  struct Params {
    const VulkanDevice *device{nullptr};
    std::vector<BufferBinding> buffer_bindings;
    SpirvCodeView code;
    std::string name{"Pipeline"};
  };

  explicit VulkanPipeline(const Params &params);
  ~VulkanPipeline();

  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }
  VkPipeline pipeline() const {
    return pipeline_;
  }
  const VkDescriptorSet &descriptor_set() const {
    return descriptor_set_;
  }
  const std::string &name() const {
    return name_;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_compute_pipeline(const Params &params);
  void create_descriptor_pool(const Params &params);
  void create_descriptor_sets(const Params &params);

  static VkShaderModule create_shader_module(VkDevice device,
                                             const SpirvCodeView &code);

  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  // TODO: Commands using the same Taichi buffers should be able to share the
  // same descriptor set layout?
  VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
  // TODO: Commands having the same |descriptor_set_layout_| should be able to
  // share the same pipeline layout?
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
  // This maps 1:1 to a shader, so it needs to be created per compute
  // shader.
  VkPipeline pipeline_{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set_{VK_NULL_HANDLE};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VkDevice device,
                    VkCommandBuffer buffer);
  ~VulkanCommandList();

  void bind_pipeline(Pipeline *p) override;
  void bind_resources(ResourceBinder &binder) override;
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
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi