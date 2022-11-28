#include <unordered_set>
#include "taichi/rhi/vulkan/vulkan_pipeline.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "spirv_reflect.h"

namespace taichi::lang {
namespace vulkan {

VulkanPipeline::VulkanPipeline(const Params &params)
    : device_(params.device->vk_device()), name_(params.name) {
  create_descriptor_set_layout(params);
  create_shader_stages(params);
  create_pipeline_layout();
  create_compute_pipeline(params);

  for (VkShaderModule shader_module : shader_modules_) {
    vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
  }
  shader_modules_.clear();
}

VulkanPipeline::VulkanPipeline(
    const Params &params,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs)
    : device_(params.device->vk_device()), name_(params.name) {
  this->graphics_pipeline_template_ =
      std::make_unique<GraphicsPipelineTemplate>();

  create_descriptor_set_layout(params);
  create_shader_stages(params);
  create_pipeline_layout();
  create_graphics_pipeline(raster_params, vertex_inputs, vertex_attrs);
}

VulkanPipeline::~VulkanPipeline() {
  for (VkShaderModule shader_module : shader_modules_) {
    vkDestroyShaderModule(device_, shader_module, kNoVkAllocCallbacks);
  }
  shader_modules_.clear();
}

VkShaderModule VulkanPipeline::create_shader_module(VkDevice device,
                                                    const SpirvCodeView &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size;
  create_info.pCode = code.data;

  VkShaderModule shader_module;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateShaderModule(device, &create_info, kNoVkAllocCallbacks,
                           &shader_module),
      "failed to create shader module");
  return shader_module;
}

vkapi::IVkPipeline VulkanPipeline::graphics_pipeline(
    const VulkanRenderPassDesc &renderpass_desc,
    vkapi::IVkRenderPass renderpass) {
  if (graphics_pipeline_.find(renderpass) != graphics_pipeline_.end()) {
    return graphics_pipeline_.at(renderpass);
  }

  vkapi::IVkPipeline pipeline = vkapi::create_graphics_pipeline(
      device_, &graphics_pipeline_template_->pipeline_info, renderpass,
      pipeline_layout_);

  graphics_pipeline_[renderpass] = pipeline;

  return pipeline;
}

void VulkanPipeline::create_descriptor_set_layout(const Params &params) {
  std::unordered_set<uint32_t> sets_used;

  for (auto &code_view : params.code) {
    SpvReflectShaderModule module;
    SpvReflectResult result =
        spvReflectCreateShaderModule(code_view.size, code_view.data, &module);
    TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

    uint32_t set_count = 0;
    result = spvReflectEnumerateDescriptorSets(&module, &set_count, nullptr);
    TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
    std::vector<SpvReflectDescriptorSet *> desc_sets(set_count);
    result = spvReflectEnumerateDescriptorSets(&module, &set_count,
                                               desc_sets.data());
    TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

    for (SpvReflectDescriptorSet *desc_set : desc_sets) {
      uint32_t set = desc_set->set;
      for (int i = 0; i < desc_set->binding_count; i++) {
        SpvReflectDescriptorBinding *desc_binding = desc_set->bindings[i];

        if (desc_binding->descriptor_type ==
            SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          resource_binder_.rw_buffer(set, desc_binding->binding, kDeviceNullPtr,
                                     0);
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
          resource_binder_.buffer(set, desc_binding->binding, kDeviceNullPtr,
                                  0);
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
          resource_binder_.image(set, desc_binding->binding,
                                 kDeviceNullAllocation, {});
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
          resource_binder_.rw_image(set, desc_binding->binding,
                                    kDeviceNullAllocation, {});
        } else {
          TI_WARN("unrecognized binding");
        }
      }
      sets_used.insert(set);
    }

    // Handle special vertex shaders stuff
    // if (code_view.stage == VK_SHADER_STAGE_VERTEX_BIT) {
    //   uint32_t attrib_count;
    //   result =
    //       spvReflectEnumerateInputVariables(&module, &attrib_count, nullptr);
    //   TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
    //   std::vector<SpvReflectInterfaceVariable *> attribs(attrib_count);
    //   result = spvReflectEnumerateInputVariables(&module, &attrib_count,
    //                                               attribs.data());
    //   TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

    //   for (SpvReflectInterfaceVariable *attrib : attribs) {
    //     uint32_t location = attrib->location;
    //     SpvReflectTypeDescription *type = attrib->type_description;
    //     TI_WARN("attrib {}:{}", location, type->type_name);
    //   }
    // }

    if (code_view.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
      uint32_t render_target_count = 0;
      result = spvReflectEnumerateOutputVariables(&module, &render_target_count,
                                                  nullptr);
      TI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

      std::vector<SpvReflectInterfaceVariable *> variables(render_target_count);
      result = spvReflectEnumerateOutputVariables(&module, &render_target_count,
                                                  variables.data());

      render_target_count = 0;

      for (auto var : variables) {
        // We want to remove auxiliary outputs such as frag depth
        if (static_cast<int>(var->built_in) == -1) {
          render_target_count++;
        }
      }

      graphics_pipeline_template_->blend_attachments.resize(
          render_target_count);

      VkPipelineColorBlendAttachmentState default_state{};
      default_state.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      default_state.blendEnable = VK_FALSE;

      std::fill(graphics_pipeline_template_->blend_attachments.begin(),
                graphics_pipeline_template_->blend_attachments.end(),
                default_state);
    }
    spvReflectDestroyShaderModule(&module);
  }

  for (uint32_t set : sets_used) {
    vkapi::IVkDescriptorSetLayout layout =
        params.device->get_desc_set_layout(resource_binder_.get_set(set));

    set_layouts_.push_back(layout);
  }

  resource_binder_.lock_layout();
}

void VulkanPipeline::create_shader_stages(const Params &params) {
  for (auto &code_view : params.code) {
    VkPipelineShaderStageCreateInfo &shader_stage_info =
        shader_stages_.emplace_back();

    VkShaderModule shader_module = create_shader_module(device_, code_view);

    shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = code_view.stage;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = "main";

    shader_modules_.push_back(shader_module);
  }
}

void VulkanPipeline::create_pipeline_layout() {
  pipeline_layout_ = vkapi::create_pipeline_layout(device_, set_layouts_);
}

void VulkanPipeline::create_compute_pipeline(const Params &params) {
  TI_TRACE("Compiling Vulkan pipeline {}", params.name);
  pipeline_ = vkapi::create_compute_pipeline(device_, 0, shader_stages_[0],
                                             pipeline_layout_);
}

void VulkanPipeline::create_graphics_pipeline(
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs) {
  // Use dynamic viewport state. These two are just dummies
  VkViewport viewport;
  viewport.width = 1;
  viewport.height = 1;
  viewport.x = 0;
  viewport.y = 0;
  viewport.minDepth = 0.0;
  viewport.maxDepth = 1.0;

  VkRect2D scissor;
  scissor.offset = {0, 0};
  scissor.extent = {1, 1};

  VkPipelineViewportStateCreateInfo &viewport_state =
      graphics_pipeline_template_->viewport_state;
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  for (const VertexInputBinding &binding : vertex_inputs) {
    VkVertexInputBindingDescription &desc =
        graphics_pipeline_template_->input_bindings.emplace_back();
    desc.binding = binding.binding;
    desc.stride = binding.stride;
    desc.inputRate = binding.instance ? VK_VERTEX_INPUT_RATE_INSTANCE
                                      : VK_VERTEX_INPUT_RATE_VERTEX;
  }

  for (const VertexInputAttribute &attr : vertex_attrs) {
    VkVertexInputAttributeDescription &desc =
        graphics_pipeline_template_->input_attrs.emplace_back();
    desc.binding = attr.binding;
    desc.location = attr.location;
    desc.format = buffer_format_ti_to_vk(attr.format);
    desc.offset = attr.offset;
  }

  VkPipelineVertexInputStateCreateInfo &vertex_input =
      graphics_pipeline_template_->input;
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.pNext = nullptr;
  vertex_input.flags = 0;
  vertex_input.vertexBindingDescriptionCount =
      graphics_pipeline_template_->input_bindings.size();
  vertex_input.pVertexBindingDescriptions =
      graphics_pipeline_template_->input_bindings.data();
  vertex_input.vertexAttributeDescriptionCount =
      graphics_pipeline_template_->input_attrs.size();
  vertex_input.pVertexAttributeDescriptions =
      graphics_pipeline_template_->input_attrs.data();

  VkPipelineInputAssemblyStateCreateInfo &input_assembly =
      graphics_pipeline_template_->input_assembly;
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  static const std::unordered_map<TopologyType, VkPrimitiveTopology>
      topo_types = {
          {TopologyType::Triangles, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
          {TopologyType::Lines, VK_PRIMITIVE_TOPOLOGY_LINE_LIST},
          {TopologyType::Points, VK_PRIMITIVE_TOPOLOGY_POINT_LIST},
      };
  input_assembly.topology = topo_types.at(raster_params.prim_topology);
  input_assembly.primitiveRestartEnable = VK_FALSE;

  static const std::unordered_map<PolygonMode, VkPolygonMode> polygon_modes = {
      {PolygonMode::Fill, VK_POLYGON_MODE_FILL},
      {PolygonMode::Line, VK_POLYGON_MODE_LINE},
      {PolygonMode::Point, VK_POLYGON_MODE_POINT},
  };

  VkPipelineRasterizationStateCreateInfo &rasterizer =
      graphics_pipeline_template_->rasterizer;
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = polygon_modes.at(raster_params.polygon_mode);
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = 0;
  if (raster_params.front_face_cull) {
    rasterizer.cullMode |= VK_CULL_MODE_FRONT_BIT;
  }
  if (raster_params.back_face_cull) {
    rasterizer.cullMode |= VK_CULL_MODE_BACK_BIT;
  }
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo &multisampling =
      graphics_pipeline_template_->multisampling;
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo &depth_stencil =
      graphics_pipeline_template_->depth_stencil;
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = raster_params.depth_test;
  depth_stencil.depthWriteEnable = raster_params.depth_write;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo &color_blending =
      graphics_pipeline_template_->color_blending;
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount =
      graphics_pipeline_template_->blend_attachments.size();
  color_blending.pAttachments =
      graphics_pipeline_template_->blend_attachments.data();
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  if (raster_params.blending.size()) {
    TI_ASSERT_INFO(raster_params.blending.size() ==
                       graphics_pipeline_template_->blend_attachments.size(),
                   "RasterParams::blending (size={}) must either be zero sized "
                   "or match the number of fragment shader outputs (size={}).",
                   raster_params.blending.size(),
                   graphics_pipeline_template_->blend_attachments.size());

    for (int i = 0; i < raster_params.blending.size(); i++) {
      auto &state = graphics_pipeline_template_->blend_attachments[i];
      auto &ti_param = raster_params.blending[i];
      state.blendEnable = ti_param.enable;
      if (ti_param.enable) {
        state.colorBlendOp = blend_op_ti_to_vk(ti_param.color.op);
        state.srcColorBlendFactor =
            blend_factor_ti_to_vk(ti_param.color.src_factor);
        state.dstColorBlendFactor =
            blend_factor_ti_to_vk(ti_param.color.dst_factor);
        state.alphaBlendOp = blend_op_ti_to_vk(ti_param.alpha.op);
        state.srcAlphaBlendFactor =
            blend_factor_ti_to_vk(ti_param.alpha.src_factor);
        state.dstAlphaBlendFactor =
            blend_factor_ti_to_vk(ti_param.alpha.dst_factor);
        state.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      }
    }
  }

  VkPipelineDynamicStateCreateInfo &dynamic_state =
      graphics_pipeline_template_->dynamic_state;
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.pNext = nullptr;
  dynamic_state.pDynamicStates =
      graphics_pipeline_template_->dynamic_state_enables.data();
  dynamic_state.dynamicStateCount =
      graphics_pipeline_template_->dynamic_state_enables.size();

  VkGraphicsPipelineCreateInfo &pipeline_info =
      graphics_pipeline_template_->pipeline_info;
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = shader_stages_.size();
  pipeline_info.pStages = shader_stages_.data();
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.renderPass = VK_NULL_HANDLE;  // Filled in later
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
}

VulkanResourceBinder::VulkanResourceBinder(VkPipelineBindPoint bind_point)
    : bind_point_(bind_point) {
}

VulkanResourceBinder::~VulkanResourceBinder() {
  for (auto &set_pair : sets_) {
    Set &set = set_pair.second;
    for (auto &binding_pair : set.bindings) {
      VkSampler sampler = binding_pair.second.sampler;
      if (sampler != VK_NULL_HANDLE) {
        Device *dev = binding_pair.second.ptr.device;
        vkDestroySampler(static_cast<VulkanDevice *>(dev)->vk_device(), sampler,
                         kNoVkAllocCallbacks);
      }
    }
  }
}

std::unique_ptr<ResourceBinder::Bindings> VulkanResourceBinder::materialize() {
  return std::unique_ptr<Bindings>();
}

} // namespace vulkan
} // namespace taichi::lang
