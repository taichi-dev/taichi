#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang {
namespace vulkan {

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

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

// VulkanPipeline maps to a vkapi::IVkPipeline, or a SPIR-V module (a GLSL
// compute shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    std::vector<SpirvCodeView> code;
    std::string name{"Pipeline"};
  };

  explicit VulkanPipeline(const Params &params);
  explicit VulkanPipeline(
      const Params &params,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);
  ~VulkanPipeline() override;

  ResourceBinder *resource_binder() override {
    return &resource_binder_;
  }

  vkapi::IVkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }

  vkapi::IVkPipeline pipeline() const {
    return pipeline_;
  }

  vkapi::IVkPipeline graphics_pipeline(
      const VulkanRenderPassDesc &renderpass_desc,
      vkapi::IVkRenderPass renderpass);

  const std::string &name() const {
    return name_;
  }

  bool is_graphics() const {
    return graphics_pipeline_template_ != nullptr;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_shader_stages(const Params &params);
  void create_pipeline_layout();
  void create_compute_pipeline(const Params &params);
  void create_graphics_pipeline(
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);

  static VkShaderModule create_shader_module(VkDevice device,
                                             const SpirvCodeView &code);

  struct GraphicsPipelineTemplate {
    VkPipelineViewportStateCreateInfo viewport_state{};
    std::vector<VkVertexInputBindingDescription> input_bindings;
    std::vector<VkVertexInputAttributeDescription> input_attrs;
    VkPipelineVertexInputStateCreateInfo input{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineColorBlendStateCreateInfo color_blending{};
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments{};
    std::vector<VkDynamicState> dynamic_state_enables = {
        VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    VkGraphicsPipelineCreateInfo pipeline_info{};
  };

  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages_;

  std::unique_ptr<GraphicsPipelineTemplate> graphics_pipeline_template_;
  std::unordered_map<vkapi::IVkRenderPass, vkapi::IVkPipeline>
      graphics_pipeline_;

  VulkanResourceBinder resource_binder_;
  std::vector<vkapi::IVkDescriptorSetLayout> set_layouts_;
  std::vector<VkShaderModule> shader_modules_;
  vkapi::IVkPipeline pipeline_{VK_NULL_HANDLE};
  vkapi::IVkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
};

} // namespace vulkan
} // namespace taichi::lang
