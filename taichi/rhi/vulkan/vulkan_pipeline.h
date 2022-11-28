#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"
// FIXME: (penguinliong) Remove this.
#include "taichi/rhi/vulkan/vulkan_resource_binder.h"

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

}  // namespace vulkan
}  // namespace taichi::lang
