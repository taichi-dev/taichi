#include "taichi/rhi/vulkan/vulkan_device_creator.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <array>
#include <set>

#include "taichi/rhi/vulkan/vulkan_common.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/common/logging.h"

#include "spirv_reflect.h"

namespace taichi {
namespace lang {
namespace vulkan {

const std::unordered_map<BufferFormat, VkFormat> buffer_format_ti_2_vk = {
    {BufferFormat::r8, VK_FORMAT_R8_UNORM},
    {BufferFormat::rg8, VK_FORMAT_R8G8_UNORM},
    {BufferFormat::rgba8, VK_FORMAT_R8G8B8A8_UNORM},
    {BufferFormat::rgba8srgb, VK_FORMAT_R8G8B8A8_SRGB},
    {BufferFormat::bgra8, VK_FORMAT_B8G8R8A8_UNORM},
    {BufferFormat::bgra8srgb, VK_FORMAT_B8G8R8A8_SRGB},
    {BufferFormat::r8u, VK_FORMAT_R8_UINT},
    {BufferFormat::rg8u, VK_FORMAT_R8G8_UINT},
    {BufferFormat::rgba8u, VK_FORMAT_R8G8B8A8_UINT},
    {BufferFormat::r8i, VK_FORMAT_R8_SINT},
    {BufferFormat::rg8i, VK_FORMAT_R8G8_SINT},
    {BufferFormat::rgba8i, VK_FORMAT_R8G8B8A8_SINT},
    {BufferFormat::r16, VK_FORMAT_R16_UNORM},
    {BufferFormat::rg16, VK_FORMAT_R16G16_UNORM},
    {BufferFormat::rgb16, VK_FORMAT_R16G16B16_UNORM},
    {BufferFormat::rgba16, VK_FORMAT_R16G16B16A16_UNORM},
    {BufferFormat::r16u, VK_FORMAT_R16_UNORM},
    {BufferFormat::rg16u, VK_FORMAT_R16G16_UNORM},
    {BufferFormat::rgb16u, VK_FORMAT_R16G16B16_UNORM},
    {BufferFormat::rgba16u, VK_FORMAT_R16G16B16A16_UNORM},
    {BufferFormat::r16i, VK_FORMAT_R16_SINT},
    {BufferFormat::rg16i, VK_FORMAT_R16G16_SINT},
    {BufferFormat::rgb16i, VK_FORMAT_R16G16B16_SINT},
    {BufferFormat::rgba16i, VK_FORMAT_R16G16B16A16_SINT},
    {BufferFormat::r16f, VK_FORMAT_R16_SFLOAT},
    {BufferFormat::rg16f, VK_FORMAT_R16G16_SFLOAT},
    {BufferFormat::rgb16f, VK_FORMAT_R16G16B16_SFLOAT},
    {BufferFormat::rgba16f, VK_FORMAT_R16G16B16A16_SFLOAT},
    {BufferFormat::r32u, VK_FORMAT_R32_UINT},
    {BufferFormat::rg32u, VK_FORMAT_R32G32_UINT},
    {BufferFormat::rgb32u, VK_FORMAT_R32G32B32_UINT},
    {BufferFormat::rgba32u, VK_FORMAT_R32G32B32A32_UINT},
    {BufferFormat::r32i, VK_FORMAT_R32_SINT},
    {BufferFormat::rg32i, VK_FORMAT_R32G32_SINT},
    {BufferFormat::rgb32i, VK_FORMAT_R32G32B32_SINT},
    {BufferFormat::rgba32i, VK_FORMAT_R32G32B32A32_SINT},
    {BufferFormat::r32f, VK_FORMAT_R32_SFLOAT},
    {BufferFormat::rg32f, VK_FORMAT_R32G32_SFLOAT},
    {BufferFormat::rgb32f, VK_FORMAT_R32G32B32_SFLOAT},
    {BufferFormat::rgba32f, VK_FORMAT_R32G32B32A32_SFLOAT},
    {BufferFormat::depth16, VK_FORMAT_D16_UNORM},
    {BufferFormat::depth24stencil8, VK_FORMAT_D24_UNORM_S8_UINT},
    {BufferFormat::depth32f, VK_FORMAT_D32_SFLOAT}};

VkFormat buffer_format_ti_to_vk(BufferFormat f) {
  if (buffer_format_ti_2_vk.find(f) == buffer_format_ti_2_vk.end()) {
    TI_ERROR("BufferFormat cannot be mapped to vk");
  }
  return buffer_format_ti_2_vk.at(f);
}

BufferFormat buffer_format_vk_to_ti(VkFormat f) {
  std::unordered_map<VkFormat, BufferFormat> inverse;
  for (auto kv : buffer_format_ti_2_vk) {
    inverse[kv.second] = kv.first;
  }
  if (inverse.find(f) == inverse.end()) {
    TI_ERROR("VkFormat cannot be mapped to ti");
  }
  return inverse.at(f);
}

const std::unordered_map<ImageLayout, VkImageLayout> image_layout_ti_2_vk = {
    {ImageLayout::undefined, VK_IMAGE_LAYOUT_UNDEFINED},
    {ImageLayout::shader_read, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    {ImageLayout::shader_write, VK_IMAGE_LAYOUT_GENERAL},
    {ImageLayout::shader_read_write, VK_IMAGE_LAYOUT_GENERAL},
    {ImageLayout::color_attachment, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    {ImageLayout::color_attachment_read,
     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    {ImageLayout::depth_attachment, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL},
    {ImageLayout::depth_attachment_read,
     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL},
    {ImageLayout::transfer_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
    {ImageLayout::transfer_src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
    {ImageLayout::present_src, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}};

VkImageLayout image_layout_ti_to_vk(ImageLayout layout) {
  if (image_layout_ti_2_vk.find(layout) == image_layout_ti_2_vk.end()) {
    TI_ERROR("ImageLayout cannot be mapped to vk");
  }
  return image_layout_ti_2_vk.at(layout);
}

const std::unordered_map<BlendOp, VkBlendOp> blend_op_ti_2_vk = {
    {BlendOp::add, VK_BLEND_OP_ADD},
    {BlendOp::subtract, VK_BLEND_OP_SUBTRACT},
    {BlendOp::reverse_subtract, VK_BLEND_OP_REVERSE_SUBTRACT},
    {BlendOp::min, VK_BLEND_OP_MIN},
    {BlendOp::max, VK_BLEND_OP_MAX}};

VkBlendOp blend_op_ti_to_vk(BlendOp op) {
  if (blend_op_ti_2_vk.find(op) == blend_op_ti_2_vk.end()) {
    TI_ERROR("BlendOp cannot be mapped to vk");
  }
  return blend_op_ti_2_vk.at(op);
}

const std::unordered_map<BlendFactor, VkBlendFactor> blend_factor_ti_2_vk = {
    {BlendFactor::zero, VK_BLEND_FACTOR_ZERO},
    {BlendFactor::one, VK_BLEND_FACTOR_ONE},
    {BlendFactor::src_color, VK_BLEND_FACTOR_SRC_COLOR},
    {BlendFactor::one_minus_src_color, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR},
    {BlendFactor::dst_color, VK_BLEND_FACTOR_DST_COLOR},
    {BlendFactor::one_minus_dst_color, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR},
    {BlendFactor::src_alpha, VK_BLEND_FACTOR_SRC_ALPHA},
    {BlendFactor::one_minus_src_alpha, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA},
    {BlendFactor::dst_alpha, VK_BLEND_FACTOR_DST_ALPHA},
    {BlendFactor::one_minus_dst_alpha, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA},
};

VkBlendFactor blend_factor_ti_to_vk(BlendFactor factor) {
  if (blend_factor_ti_2_vk.find(factor) == blend_factor_ti_2_vk.end()) {
    TI_ERROR("BlendFactor cannot be mapped to vk");
  }
  return blend_factor_ti_2_vk.at(factor);
}

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
    desc.format = buffer_format_ti_2_vk.at(attr.format);
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
  dynamic_state.pNext = NULL;
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

VkSampler create_sampler(ImageSamplerConfig config, VkDevice device) {
  VkSampler sampler = VK_NULL_HANDLE;

  // todo: fill these using the information from the ImageSamplerConfig
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
  return sampler;
}

#define CHECK_SET_BINDINGS                                          \
  bool set_not_found = (sets_.find(set) == sets_.end());            \
  if (set_not_found) {                                              \
    if (layout_locked_) {                                           \
      return;                                                       \
    } else {                                                        \
      sets_[set] = {};                                              \
    }                                                               \
  }                                                                 \
  auto &bindings = sets_.at(set).bindings;                          \
  if (layout_locked_ && bindings.find(binding) == bindings.end()) { \
    return;                                                         \
  }

void VulkanResourceBinder::rw_buffer(uint32_t set,
                                     uint32_t binding,
                                     DevicePtr ptr,
                                     size_t size) {
  CHECK_SET_BINDINGS;

  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }

  Binding new_binding = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ptr, size};
  bindings[binding] = new_binding;
}

void VulkanResourceBinder::rw_buffer(uint32_t set,
                                     uint32_t binding,
                                     DeviceAllocation alloc) {
  rw_buffer(set, binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

void VulkanResourceBinder::buffer(uint32_t set,
                                  uint32_t binding,
                                  DevicePtr ptr,
                                  size_t size) {
  CHECK_SET_BINDINGS;

  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }

  Binding new_binding = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ptr, size};
  bindings[binding] = new_binding;
}

void VulkanResourceBinder::buffer(uint32_t set,
                                  uint32_t binding,
                                  DeviceAllocation alloc) {
  buffer(set, binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

void VulkanResourceBinder::image(uint32_t set,
                                 uint32_t binding,
                                 DeviceAllocation alloc,
                                 ImageSamplerConfig sampler_config) {
  CHECK_SET_BINDINGS
  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type ==
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }
  if (bindings[binding].sampler != VK_NULL_HANDLE) {
    Device *dev = bindings[binding].ptr.device;
    vkDestroySampler(static_cast<VulkanDevice *>(dev)->vk_device(),
                     bindings[binding].sampler, kNoVkAllocCallbacks);
  }
  bindings[binding] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       alloc.get_ptr(0), VK_WHOLE_SIZE};
  if (alloc.device) {
    VulkanDevice *device = static_cast<VulkanDevice *>(alloc.device);
    bindings[binding].sampler =
        create_sampler(sampler_config, device->vk_device());
  }
}

void VulkanResourceBinder::rw_image(uint32_t set,
                                    uint32_t binding,
                                    DeviceAllocation alloc,
                                    int lod) {
  CHECK_SET_BINDINGS
  if (layout_locked_) {
    TI_ASSERT(bindings.at(binding).type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
  } else {
    if (bindings.find(binding) != bindings.end()) {
      TI_WARN("Overriding last binding");
    }
  }
  bindings[binding] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, alloc.get_ptr(0),
                       VK_WHOLE_SIZE};
}

#undef CHECK_SET_BINDINGS

void VulkanResourceBinder::vertex_buffer(DevicePtr ptr, uint32_t binding) {
  vertex_buffers_[binding] = ptr;
}

void VulkanResourceBinder::index_buffer(DevicePtr ptr, size_t index_width) {
  index_buffer_ = ptr;
  if (index_width == 32) {
    index_type_ = VK_INDEX_TYPE_UINT32;
  } else if (index_width == 16) {
    index_type_ = VK_INDEX_TYPE_UINT16;
  } else {
    TI_ERROR("unsupported index width");
  }
}

void VulkanResourceBinder::write_to_set(uint32_t index,
                                        VulkanDevice &device,
                                        vkapi::IVkDescriptorSet set) {
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  std::vector<VkDescriptorImageInfo> image_infos;
  std::vector<bool> is_image;
  std::vector<VkWriteDescriptorSet> desc_writes;

  for (auto &pair : sets_.at(index).bindings) {
    uint32_t binding = pair.first;

    if (pair.second.ptr != kDeviceNullPtr) {
      VkDescriptorBufferInfo &buffer_info = buffer_infos.emplace_back();
      VkDescriptorImageInfo &image_info = image_infos.emplace_back();

      if (pair.second.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          pair.second.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
        auto buffer = device.get_vkbuffer(pair.second.ptr);
        buffer_info.buffer = buffer->buffer;
        buffer_info.offset = pair.second.ptr.offset;
        buffer_info.range = pair.second.size;
        is_image.push_back(false);
        set->ref_binding_objs[binding] = buffer;
      } else if (pair.second.type ==
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        auto view = std::get<1>(device.get_vk_image(pair.second.ptr));
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = view->view;
        image_info.sampler = pair.second.sampler;
        is_image.push_back(true);
        set->ref_binding_objs[binding] = view;
      } else if (pair.second.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
        auto view =
            device.get_vk_lod_imageview(pair.second.ptr, pair.second.image_lod);
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_info.imageView = view->view;
        is_image.push_back(true);
        set->ref_binding_objs[binding] = view;
      } else {
        TI_NOT_IMPLEMENTED;
      }

      VkWriteDescriptorSet &write = desc_writes.emplace_back();
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.pNext = nullptr;
      write.dstSet = set->set;
      write.dstBinding = binding;
      write.dstArrayElement = 0;
      write.descriptorCount = 1;
      write.descriptorType = pair.second.type;
      write.pImageInfo = nullptr;
      write.pBufferInfo = nullptr;
      write.pTexelBufferView = nullptr;
    }
  }

  // Set these pointers later as std::vector resize can relocate the pointers
  int i = 0;
  for (auto &write : desc_writes) {
    if (is_image[i]) {
      write.pImageInfo = &image_infos[i];
    } else {
      write.pBufferInfo = &buffer_infos[i];
    }
    i++;
  }

  vkUpdateDescriptorSets(device.vk_device(), desc_writes.size(),
                         desc_writes.data(), /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);
}

void VulkanResourceBinder::lock_layout() {
  layout_locked_ = true;
}

VulkanCommandList::VulkanCommandList(VulkanDevice *ti_device,
                                     VulkanStream *stream,
                                     vkapi::IVkCommandBuffer buffer)
    : ti_device_(ti_device),
      stream_(stream),
      device_(ti_device->vk_device()),
      query_pool_(vkapi::create_query_pool(ti_device->vk_device())),
      buffer_(buffer) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  vkBeginCommandBuffer(buffer->buffer, &info);
  vkCmdResetQueryPool(buffer->buffer, query_pool_->query_pool, 0, 2);
  vkCmdWriteTimestamp(buffer->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      query_pool_->query_pool, 0);
}

VulkanCommandList::~VulkanCommandList() {
}

void VulkanCommandList::bind_pipeline(Pipeline *p) {
  auto pipeline = static_cast<VulkanPipeline *>(p);

  if (current_pipeline_ == pipeline)
    return;

  if (pipeline->is_graphics()) {
    vkapi::IVkPipeline vk_pipeline = pipeline->graphics_pipeline(
        current_renderpass_desc_, current_renderpass_);
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vk_pipeline->pipeline);

    VkViewport viewport;
    viewport.width = viewport_width_;
    viewport.height = viewport_height_;
    viewport.x = 0;
    viewport.y = 0;
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    VkRect2D scissor;
    scissor.offset = {0, 0};
    scissor.extent = {viewport_width_, viewport_height_};

    vkCmdSetViewport(buffer_->buffer, 0, 1, &viewport);
    vkCmdSetScissor(buffer_->buffer, 0, 1, &scissor);
    vkCmdSetLineWidth(buffer_->buffer, 1.0f);
    buffer_->refs.push_back(vk_pipeline);
  } else {
    auto vk_pipeline = pipeline->pipeline();
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      vk_pipeline->pipeline);
    buffer_->refs.push_back(vk_pipeline);
  }

  current_pipeline_ = pipeline;
}

void VulkanCommandList::bind_resources(ResourceBinder *ti_binder) {
  VulkanResourceBinder *binder = static_cast<VulkanResourceBinder *>(ti_binder);

  for (auto &pair : binder->get_sets()) {
    VkPipelineLayout pipeline_layout =
        current_pipeline_->pipeline_layout()->layout;

    vkapi::IVkDescriptorSetLayout layout =
        ti_device_->get_desc_set_layout(pair.second);

    vkapi::IVkDescriptorSet set = nullptr;

    if (currently_used_sets_.find(pair.second) != currently_used_sets_.end()) {
      set = currently_used_sets_.at(pair.second);
    }

    if (!set) {
      set = ti_device_->alloc_desc_set(layout);
      binder->write_to_set(pair.first, *ti_device_, set);
      currently_used_sets_[pair.second] = set;
    }

    VkPipelineBindPoint bind_point;
    if (current_pipeline_->is_graphics()) {
      bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    } else {
      bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    }

    vkCmdBindDescriptorSets(buffer_->buffer, bind_point, pipeline_layout,
                            /*firstSet=*/0,
                            /*descriptorSetCount=*/1, &set->set,
                            /*dynamicOffsetCount=*/0,
                            /*pDynamicOffsets=*/nullptr);
    buffer_->refs.push_back(set);
  }

  if (current_pipeline_->is_graphics()) {
    auto [idx_ptr, type] = binder->get_index_buffer();
    if (idx_ptr.device) {
      auto index_buffer = ti_device_->get_vkbuffer(idx_ptr);
      vkCmdBindIndexBuffer(buffer_->buffer, index_buffer->buffer,
                           idx_ptr.offset, type);
      buffer_->refs.push_back(index_buffer);
    }

    for (auto [binding, ptr] : binder->get_vertex_buffers()) {
      auto buffer = ti_device_->get_vkbuffer(ptr);
      vkCmdBindVertexBuffers(buffer_->buffer, binding, 1, &buffer->buffer,
                             &ptr.offset);
      buffer_->refs.push_back(buffer);
    }
  }
}

void VulkanCommandList::bind_resources(ResourceBinder *binder,
                                       ResourceBinder::Bindings *bindings) {
}

void VulkanCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  TI_ASSERT(ptr.device == ti_device_);

  auto buffer = ti_device_->get_vkbuffer(ptr);

  VkBufferMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.buffer = buffer->buffer;
  barrier.offset = ptr.offset;
  barrier.size = size;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_->buffer,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/0, nullptr,
      /*bufferMemoryBarrierCount=*/1,
      /*pBufferMemoryBarriers=*/&barrier,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::buffer_barrier(DeviceAllocation alloc) {
  buffer_barrier(DevicePtr{alloc, 0}, VK_WHOLE_SIZE);
}

void VulkanCommandList::memory_barrier() {
  VkMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_->buffer,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/1, &barrier,
      /*bufferMemoryBarrierCount=*/0,
      /*pBufferMemoryBarriers=*/nullptr,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
}

void VulkanCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  VkBufferCopy copy_region{};
  copy_region.srcOffset = src.offset;
  copy_region.dstOffset = dst.offset;
  copy_region.size = size;
  auto src_buffer = ti_device_->get_vkbuffer(src);
  auto dst_buffer = ti_device_->get_vkbuffer(dst);
  vkCmdCopyBuffer(buffer_->buffer, src_buffer->buffer, dst_buffer->buffer,
                  /*regionCount=*/1, &copy_region);
  buffer_->refs.push_back(src_buffer);
  buffer_->refs.push_back(dst_buffer);
}

void VulkanCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  auto buffer = ti_device_->get_vkbuffer(ptr);
  vkCmdFillBuffer(buffer_->buffer, buffer->buffer, ptr.offset,
                  (size == kBufferSizeEntireSize) ? VK_WHOLE_SIZE : size, data);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  vkCmdDispatch(buffer_->buffer, x, y, z);
}

vkapi::IVkCommandBuffer VulkanCommandList::vk_command_buffer() {
  return buffer_;
}

vkapi::IVkQueryPool VulkanCommandList::vk_query_pool() {
  return query_pool_;
}

void VulkanCommandList::begin_renderpass(int x0,
                                         int y0,
                                         int x1,
                                         int y1,
                                         uint32_t num_color_attachments,
                                         DeviceAllocation *color_attachments,
                                         bool *color_clear,
                                         std::vector<float> *clear_colors,
                                         DeviceAllocation *depth_attachment,
                                         bool depth_clear) {
  VulkanRenderPassDesc &rp_desc = current_renderpass_desc_;
  current_renderpass_desc_.color_attachments.clear();
  rp_desc.clear_depth = depth_clear;

  bool has_depth = false;

  if (depth_attachment) {
    auto [image, view, format] = ti_device_->get_vk_image(*depth_attachment);
    rp_desc.depth_attachment = format;
    has_depth = true;
  } else {
    rp_desc.depth_attachment = VK_FORMAT_UNDEFINED;
  }

  std::vector<VkClearValue> clear_values(num_color_attachments +
                                         (has_depth ? 1 : 0));

  VulkanFramebufferDesc fb_desc;

  for (uint32_t i = 0; i < num_color_attachments; i++) {
    auto [image, view, format] = ti_device_->get_vk_image(color_attachments[i]);
    rp_desc.color_attachments.emplace_back(format, color_clear[i]);
    fb_desc.attachments.push_back(view);
    clear_values[i].color =
        VkClearColorValue{{clear_colors[i][0], clear_colors[i][1],
                           clear_colors[i][2], clear_colors[i][3]}};
  }

  if (has_depth) {
    auto [depth_image, depth_view, depth_format] =
        ti_device_->get_vk_image(*depth_attachment);
    clear_values[num_color_attachments].depthStencil =
        VkClearDepthStencilValue{0.0, 0};
    fb_desc.attachments.push_back(depth_view);
  }

  current_renderpass_ = ti_device_->get_renderpass(rp_desc);

  fb_desc.width = x1 - x0;
  fb_desc.height = y1 - y0;
  fb_desc.renderpass = current_renderpass_;

  viewport_width_ = fb_desc.width;
  viewport_height_ = fb_desc.height;

  current_framebuffer_ = ti_device_->get_framebuffer(fb_desc);

  VkRect2D render_area;
  render_area.offset = {x0, y0};
  render_area.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0)};

  VkRenderPassBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.renderPass = current_renderpass_->renderpass;
  begin_info.framebuffer = current_framebuffer_->framebuffer;
  begin_info.renderArea = render_area;
  begin_info.clearValueCount = clear_values.size();
  begin_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(buffer_->buffer, &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  buffer_->refs.push_back(current_renderpass_);
  buffer_->refs.push_back(current_framebuffer_);
}

void VulkanCommandList::end_renderpass() {
  vkCmdEndRenderPass(buffer_->buffer);

  current_renderpass_ = VK_NULL_HANDLE;
  current_framebuffer_ = VK_NULL_HANDLE;
}

void VulkanCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  vkCmdDraw(buffer_->buffer, num_verticies, /*instanceCount=*/1, start_vertex,
            /*firstInstance=*/0);
}

void VulkanCommandList::draw_instance(uint32_t num_verticies,
                                      uint32_t num_instances,
                                      uint32_t start_vertex,
                                      uint32_t start_instance) {
  vkCmdDraw(buffer_->buffer, num_verticies, num_instances, start_vertex,
            start_instance);
}

void VulkanCommandList::draw_indexed(uint32_t num_indicies,
                                     uint32_t start_vertex,
                                     uint32_t start_index) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, /*instanceCount=*/1,
                   start_index, start_vertex,
                   /*firstInstance=*/0);
}

void VulkanCommandList::draw_indexed_instance(uint32_t num_indicies,
                                              uint32_t num_instances,
                                              uint32_t start_vertex,
                                              uint32_t start_index,
                                              uint32_t start_instance) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, num_instances, start_index,
                   start_vertex, start_instance);
}

void VulkanCommandList::image_transition(DeviceAllocation img,
                                         ImageLayout old_layout_,
                                         ImageLayout new_layout_) {
  auto [image, view, format] = ti_device_->get_vk_image(img);

  VkImageLayout old_layout = image_layout_ti_to_vk(old_layout_);
  VkImageLayout new_layout = image_layout_ti_to_vk(new_layout_);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image->image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  static std::unordered_map<VkImageLayout, VkPipelineStageFlagBits> stages;
  stages[VK_IMAGE_LAYOUT_UNDEFINED] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_GENERAL] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stages[VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL] = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stages[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] =
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  stages[VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL] =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  stages[VK_IMAGE_LAYOUT_PRESENT_SRC_KHR] = VK_PIPELINE_STAGE_TRANSFER_BIT;

  static std::unordered_map<VkImageLayout, VkAccessFlagBits> access;
  access[VK_IMAGE_LAYOUT_UNDEFINED] = (VkAccessFlagBits)0;
  access[VK_IMAGE_LAYOUT_GENERAL] =
      VkAccessFlagBits(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  access[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_ACCESS_TRANSFER_WRITE_BIT;
  access[VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL] = VK_ACCESS_TRANSFER_READ_BIT;
  access[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] = VK_ACCESS_MEMORY_READ_BIT;
  access[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  access[VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL] =
      VkAccessFlagBits(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  access[VK_IMAGE_LAYOUT_PRESENT_SRC_KHR] = VK_ACCESS_MEMORY_READ_BIT;

  if (stages.find(old_layout) == stages.end() ||
      stages.find(new_layout) == stages.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  source_stage = stages.at(old_layout);
  destination_stage = stages.at(new_layout);

  if (access.find(old_layout) == access.end() ||
      access.find(new_layout) == access.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  barrier.srcAccessMask = access.at(old_layout);
  barrier.dstAccessMask = access.at(new_layout);

  vkCmdPipelineBarrier(buffer_->buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
  buffer_->refs.push_back(image);
}

inline void buffer_image_copy_ti_to_vk(VkBufferImageCopy &copy_info,
                                       size_t offset,
                                       const BufferImageCopyParams &params) {
  copy_info.bufferOffset = offset;
  copy_info.bufferRowLength = params.buffer_row_length;
  copy_info.bufferImageHeight = params.buffer_image_height;
  copy_info.imageExtent.width = params.image_extent.x;
  copy_info.imageExtent.height = params.image_extent.y;
  copy_info.imageExtent.depth = params.image_extent.z;
  copy_info.imageOffset.x = params.image_offset.x;
  copy_info.imageOffset.y = params.image_offset.y;
  copy_info.imageOffset.z = params.image_offset.z;
  copy_info.imageSubresource.aspectMask =
      params.image_aspect_flag;  // FIXME: add option in BufferImageCopyParams
                                 // to support copying depth images
                                 // FIXED: added an option in
                                 // BufferImageCopyParams as image_aspect_flag
                                 // by yuhaoLong(mocki)
  copy_info.imageSubresource.baseArrayLayer = params.image_base_layer;
  copy_info.imageSubresource.layerCount = params.image_layer_count;
  copy_info.imageSubresource.mipLevel = params.image_mip_level;
}

void VulkanCommandList::buffer_to_image(DeviceAllocation dst_img,
                                        DevicePtr src_buf,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, src_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(dst_img);
  auto buffer = ti_device_->get_vkbuffer(src_buf);

  vkCmdCopyBufferToImage(buffer_->buffer, buffer->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), 1, &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::image_to_buffer(DevicePtr dst_buf,
                                        DeviceAllocation src_img,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, dst_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(src_img);
  auto buffer = ti_device_->get_vkbuffer(dst_buf);

  vkCmdCopyImageToBuffer(buffer_->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), buffer->buffer, 1,
                         &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::copy_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageCopyParams &params) {
  VkImageCopy copy{};
  copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.srcSubresource.layerCount = 1;
  copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.dstSubresource.layerCount = 1;
  copy.extent.width = params.width;
  copy.extent.height = params.height;
  copy.extent.depth = params.depth;

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdCopyImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &copy);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::blit_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageCopyParams &params) {
  VkOffset3D blit_size;
  blit_size.x = params.width;
  blit_size.y = params.height;
  blit_size.z = params.depth;
  VkImageBlit blit{};
  blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.layerCount = 1;
  blit.srcOffsets[1] = blit_size;
  blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.layerCount = 1;
  blit.dstOffsets[1] = blit_size;

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdBlitImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &blit,
                 VK_FILTER_NEAREST);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::signal_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdSetEvent(buffer_->buffer, event2->vkapi_ref->event,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
void VulkanCommandList::reset_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdResetEvent(buffer_->buffer, event2->vkapi_ref->event,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
void VulkanCommandList::wait_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdWaitEvents(buffer_->buffer, 1, &event2->vkapi_ref->event,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0,
                  nullptr);
}

void VulkanCommandList::set_line_width(float width) {
  if (ti_device_->get_cap(DeviceCapability::wide_lines)) {
    vkCmdSetLineWidth(buffer_->buffer, width);
  }
}

vkapi::IVkRenderPass VulkanCommandList::current_renderpass() {
  return current_renderpass_;
}

vkapi::IVkCommandBuffer VulkanCommandList::finalize() {
  if (!finalized_) {
    vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        query_pool_->query_pool, 1);
    vkEndCommandBuffer(buffer_->buffer);
    finalized_ = true;
  }
  return buffer_;
}

struct VulkanDevice::ThreadLocalStreams {
  unordered_map<std::thread::id, std::unique_ptr<VulkanStream>> map;
};

VulkanDevice::VulkanDevice()
    : compute_streams_(std::make_unique<ThreadLocalStreams>()),
      graphics_streams_(std::make_unique<ThreadLocalStreams>()) {
}

void VulkanDevice::init_vulkan_structs(Params &params) {
  instance_ = params.instance;
  device_ = params.device;
  physical_device_ = params.physical_device;
  compute_queue_ = params.compute_queue;
  compute_queue_family_index_ = params.compute_queue_family_index;
  graphics_queue_ = params.graphics_queue;
  graphics_queue_family_index_ = params.graphics_queue_family_index;

  create_vma_allocator();
  new_descriptor_pool();
}

VulkanDevice::~VulkanDevice() {
  // Note: Ideally whoever allocated the buffer & image should be responsible
  // for deallocation as well.
  // These manual deallocations work as last resort for the case where we
  // have GGUI window whose lifetime is controlled by Python but
  // shares the same underlying VulkanDevice with Program. In an extreme
  // edge case when Python shuts down and program gets destructed before
  // GGUI Window, buffers and images allocated through GGUI window won't
  // be properly deallocated before VulkanDevice destruction. This isn't
  // the most proper fix but is less intrusive compared to other
  // approaches.
  for (auto &alloc : allocations_) {
    alloc.second.buffer.reset();
  }
  for (auto &alloc : image_allocations_) {
    alloc.second.image.reset();
  }
  allocations_.clear();
  image_allocations_.clear();

  vkDeviceWaitIdle(device_);

  desc_pool_ = nullptr;

  framebuffer_pools_.clear();
  renderpass_pools_.clear();

  vmaDestroyAllocator(allocator_);
  vmaDestroyAllocator(allocator_export_);
}

std::unique_ptr<Pipeline> VulkanDevice::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  TI_ASSERT(src.type == PipelineSourceType::spirv_binary &&
            src.stage == PipelineStageType::compute);

  SpirvCodeView code;
  code.data = (uint32_t *)src.data;
  code.size = src.size;
  code.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  VulkanPipeline::Params params;
  params.code = {code};
  params.device = this;
  params.name = name;

  return std::make_unique<VulkanPipeline>(params);
}

std::unique_ptr<DeviceEvent> VulkanDevice::create_event() {
  return std::unique_ptr<DeviceEvent>(
      new VulkanDeviceEvent(vkapi::create_event(device_, 0)));
}

// #define TI_VULKAN_DEBUG_ALLOCATIONS

DeviceAllocation VulkanDevice::allocate_memory(const AllocParams &params) {
  DeviceAllocation handle;

  handle.device = this;
  handle.alloc_id = alloc_cnt_++;

  allocations_[handle.alloc_id] = {};
  AllocationInternal &alloc = allocations_[handle.alloc_id];

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = params.size;
  // FIXME: How to express this in a backend-neutral way?
  buffer_info.usage =
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (params.usage && AllocUsage::Storage) {
    buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Uniform) {
    buffer_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Vertex) {
    buffer_info.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Index) {
    buffer_info.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    buffer_info.queueFamilyIndexCount = 2;
    buffer_info.pQueueFamilyIndices = queue_family_indices;
  }

  VkExternalMemoryBufferCreateInfo external_mem_buffer_create_info = {};
  external_mem_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_mem_buffer_create_info.pNext = NULL;

#ifdef _WIN64
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  bool export_sharing = params.export_sharing &&
                        this->get_cap(DeviceCapability::vk_has_external_memory);

  VmaAllocationCreateInfo alloc_info{};
  if (export_sharing) {
    buffer_info.pNext = &external_mem_buffer_create_info;
  }
#ifdef __APPLE__
  // weird behavior on apple: these flags are needed even if either read or
  // write is required
  if (params.host_read || params.host_write) {
#else
  if (params.host_read && params.host_write) {
#endif  //__APPLE__
    // This should be the unified memory on integrated GPUs
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
#ifdef __APPLE__
    // weird behavior on apple: if coherent bit is not set, then the memory
    // writes between map() and unmap() cannot be seen by gpu
    alloc_info.preferredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
#endif  //__APPLE__
  } else if (params.host_read) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  } else if (params.host_write) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  } else {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }

  if (get_cap(DeviceCapability::spirv_has_physical_storage_buffer)) {
    buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;
  }

  alloc.buffer = vkapi::create_buffer(
      device_, export_sharing ? allocator_export_ : allocator_, &buffer_info,
      &alloc_info);
  vmaGetAllocationInfo(alloc.buffer->allocator, alloc.buffer->allocation,
                       &alloc.alloc_info);

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  TI_TRACE("Allocate VK buffer {}, alloc_id={}", (void *)alloc.buffer,
           handle.alloc_id);
#endif

  if (get_cap(DeviceCapability::spirv_has_physical_storage_buffer)) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = alloc.buffer->buffer;
    info.pNext = nullptr;
    alloc.addr = vkGetBufferDeviceAddressKHR(device_, &info);
  }

  return handle;
}

void VulkanDevice::dealloc_memory(DeviceAllocation handle) {
  auto map_pair = allocations_.find(handle.alloc_id);

  TI_ASSERT_INFO(map_pair != allocations_.end(),
                 "Invalid handle (double free?) {}", handle.alloc_id);

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  AllocationInternal &alloc = map_pair->second;
  TI_TRACE("Dealloc VK buffer {}, alloc_id={}", (void *)alloc.buffer,
           handle.alloc_id);
#endif

  allocations_.erase(handle.alloc_id);
}

uint64_t VulkanDevice::get_memory_physical_pointer(DeviceAllocation handle) {
  const auto &alloc_int = allocations_.at(handle.alloc_id);
  return uint64_t(alloc_int.addr);
}

void *VulkanDevice::map_range(DevicePtr ptr, uint64_t size) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  if (alloc_int.buffer->allocator) {
    vmaMapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation,
                 &alloc_int.mapped);
    alloc_int.mapped = (uint8_t *)(alloc_int.mapped) + ptr.offset;
  } else {
    vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                alloc_int.alloc_info.offset + ptr.offset, size, 0,
                &alloc_int.mapped);
  }

  return alloc_int.mapped;
}

void *VulkanDevice::map(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  if (alloc_int.buffer->allocator) {
    vmaMapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation,
                 &alloc_int.mapped);
  } else {
    vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                alloc_int.alloc_info.offset, alloc_int.alloc_info.size, 0,
                &alloc_int.mapped);
  }

  return alloc_int.mapped;
}

void VulkanDevice::unmap(DevicePtr ptr) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  if (alloc_int.buffer->allocator) {
    vmaUnmapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation);
  } else {
    vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  }

  alloc_int.mapped = nullptr;
}

void VulkanDevice::unmap(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  if (alloc_int.buffer->allocator) {
    vmaUnmapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation);
  } else {
    vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  }

  alloc_int.mapped = nullptr;
}

void VulkanDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  // TODO: always create a queue specifically for transfer
  Stream *stream = get_compute_stream();
  std::unique_ptr<CommandList> cmd = stream->new_command_list();
  cmd->buffer_copy(dst, src, size);
  stream->submit_synced(cmd.get());
}

Stream *VulkanDevice::get_compute_stream() {
  auto tid = std::this_thread::get_id();
  auto &stream_map = compute_streams_->map;
  auto iter = stream_map.find(tid);
  if (iter == stream_map.end()) {
    stream_map[tid] = std::make_unique<VulkanStream>(
        *this, compute_queue_, compute_queue_family_index_);
    return stream_map.at(tid).get();
  }
  return iter->second.get();
}

Stream *VulkanDevice::get_graphics_stream() {
  auto tid = std::this_thread::get_id();
  auto &stream_map = graphics_streams_->map;
  auto iter = stream_map.find(tid);
  if (iter == stream_map.end()) {
    stream_map[tid] = std::make_unique<VulkanStream>(
        *this, graphics_queue_, graphics_queue_family_index_);
    return stream_map.at(tid).get();
  }
  return iter->second.get();
}

void VulkanDevice::wait_idle() {
  for (auto &[tid, stream] : compute_streams_->map) {
    stream->command_sync();
  }
  for (auto &[tid, stream] : graphics_streams_->map) {
    stream->command_sync();
  }
}

std::unique_ptr<CommandList> VulkanStream::new_command_list() {
  vkapi::IVkCommandBuffer buffer =
      vkapi::allocate_command_buffer(command_pool_);

  return std::make_unique<VulkanCommandList>(&device_, this, buffer);
}

StreamSemaphore VulkanStream::submit(
    CommandList *cmdlist_,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  VulkanCommandList *cmdlist = static_cast<VulkanCommandList *>(cmdlist_);
  vkapi::IVkCommandBuffer buffer = cmdlist->finalize();
  vkapi::IVkQueryPool query_pool = cmdlist->vk_query_pool();

  /*
  if (in_flight_cmdlists_.find(buffer) != in_flight_cmdlists_.end()) {
    TI_ERROR("Can not submit command list that is still in-flight");
    return;
  }
  */

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &buffer->buffer;

  std::vector<VkSemaphore> vk_wait_semaphores;
  std::vector<VkPipelineStageFlags> vk_wait_stages;

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
    vk_wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    buffer->refs.push_back(sema->vkapi_ref);
  }

  submit_info.pWaitSemaphores = vk_wait_semaphores.data();
  submit_info.waitSemaphoreCount = vk_wait_semaphores.size();
  submit_info.pWaitDstStageMask = vk_wait_stages.data();

  auto semaphore = vkapi::create_semaphore(buffer->device, 0);
  buffer->refs.push_back(semaphore);

  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &semaphore->semaphore;

  auto fence = vkapi::create_fence(buffer->device, 0);

  // Resource tracking, check previously submitted commands
  // FIXME: Figure out why it doesn't work
  /*
  std::remove_if(submitted_cmdbuffers_.begin(), submitted_cmdbuffers_.end(),
                 [&](const TrackedCmdbuf &tracked) {
                   // If fence is signaled, cmdbuf has completed
                   VkResult res =
                       vkGetFenceStatus(buffer->device, tracked.fence->fence);
                   return res == VK_SUCCESS;
    });
  */

  submitted_cmdbuffers_.push_back(TrackedCmdbuf{fence, buffer, query_pool});

  BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(queue_, /*submitCount=*/1, &submit_info,
                                      /*fence=*/fence->fence),
                        "failed to submit command buffer");

  return std::make_shared<VulkanStreamSemaphoreObject>(semaphore);
}

StreamSemaphore VulkanStream::submit_synced(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  auto sema = submit(cmdlist, wait_semaphores);
  command_sync();
  return sema;
}

void VulkanStream::command_sync() {
  vkQueueWaitIdle(queue_);

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device_.vk_physical_device(), &props);

  for (const auto &cmdbuf : submitted_cmdbuffers_) {
    if (cmdbuf.query_pool == nullptr) {
      continue;
    }

    uint64_t t[2];
    vkGetQueryPoolResults(device_.vk_device(), cmdbuf.query_pool->query_pool, 0,
                          2, sizeof(uint64_t) * 2, &t, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    double duration_us = (t[1] - t[0]) * props.limits.timestampPeriod / 1000.0;
    device_time_elapsed_us_ += duration_us;
  }

  submitted_cmdbuffers_.clear();
}

double VulkanStream::device_time_elapsed_us() const {
  return device_time_elapsed_us_;
}

std::unique_ptr<Pipeline> VulkanDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  VulkanPipeline::Params params;
  params.code = {};
  params.device = this;
  params.name = name;

  for (auto src_desc : src) {
    SpirvCodeView &code = params.code.emplace_back();
    code.data = (uint32_t *)src_desc.data;
    code.size = src_desc.size;
    code.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    if (src_desc.stage == PipelineStageType::fragment) {
      code.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (src_desc.stage == PipelineStageType::vertex) {
      code.stage = VK_SHADER_STAGE_VERTEX_BIT;
    } else if (src_desc.stage == PipelineStageType::geometry) {
      code.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_control) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_eval) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    }
  }

  return std::make_unique<VulkanPipeline>(params, raster_params, vertex_inputs,
                                          vertex_attrs);
}

std::unique_ptr<Surface> VulkanDevice::create_surface(
    const SurfaceConfig &config) {
  return std::make_unique<VulkanSurface>(this, config);
}

std::tuple<VkDeviceMemory, size_t, size_t>
VulkanDevice::get_vkmemory_offset_size(const DeviceAllocation &alloc) const {
  auto buffer_alloc = allocations_.find(alloc.alloc_id);
  if (buffer_alloc != allocations_.end()) {
    return std::make_tuple(buffer_alloc->second.alloc_info.deviceMemory,
                           buffer_alloc->second.alloc_info.offset,
                           buffer_alloc->second.alloc_info.size);
  } else {
    const ImageAllocInternal &image_alloc =
        image_allocations_.at(alloc.alloc_id);
    return std::make_tuple(image_alloc.alloc_info.deviceMemory,
                           image_alloc.alloc_info.offset,
                           image_alloc.alloc_info.size);
  }
}

vkapi::IVkBuffer VulkanDevice::get_vkbuffer(
    const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  return alloc_int.buffer;
}

std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat>
VulkanDevice::get_vk_image(const DeviceAllocation &alloc) const {
  const ImageAllocInternal &alloc_int = image_allocations_.at(alloc.alloc_id);

  return std::make_tuple(alloc_int.image, alloc_int.view,
                         alloc_int.image->format);
}

vkapi::IVkFramebuffer VulkanDevice::get_framebuffer(
    const VulkanFramebufferDesc &desc) {
  if (framebuffer_pools_.find(desc) != framebuffer_pools_.end()) {
    return framebuffer_pools_.at(desc);
  }

  vkapi::IVkFramebuffer framebuffer = vkapi::create_framebuffer(
      0, desc.renderpass, desc.attachments, desc.width, desc.height, 1);

  framebuffer_pools_.insert({desc, framebuffer});

  return framebuffer;
}

DeviceAllocation VulkanDevice::import_vkbuffer(vkapi::IVkBuffer buffer) {
  AllocationInternal alloc_int{};
  alloc_int.external = true;
  alloc_int.buffer = buffer;
  alloc_int.mapped = nullptr;
  alloc_int.addr = 0;

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_cnt_++;

  allocations_[alloc.alloc_id] = alloc_int;

  return alloc;
}

DeviceAllocation VulkanDevice::import_vk_image(vkapi::IVkImage image,
                                               vkapi::IVkImageView view,
                                               VkImageLayout layout) {
  ImageAllocInternal alloc_int;
  alloc_int.external = true;
  alloc_int.image = image;
  alloc_int.view = view;

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_cnt_++;

  image_allocations_[alloc.alloc_id] = alloc_int;

  return alloc;
}

vkapi::IVkImageView VulkanDevice::get_vk_imageview(
    const DeviceAllocation &alloc) const {
  return std::get<1>(get_vk_image(alloc));
}

vkapi::IVkImageView VulkanDevice::get_vk_lod_imageview(
    const DeviceAllocation &alloc,
    int lod) const {
  return image_allocations_.at(alloc.alloc_id).view_lods[lod];
}

DeviceAllocation VulkanDevice::create_image(const ImageParams &params) {
  DeviceAllocation handle;
  handle.device = this;
  handle.alloc_id = alloc_cnt_++;

  image_allocations_[handle.alloc_id] = {};
  ImageAllocInternal &alloc = image_allocations_[handle.alloc_id];

  int num_mip_levels = 1;

  bool is_depth = params.format == BufferFormat::depth16 ||
                  params.format == BufferFormat::depth24stencil8 ||
                  params.format == BufferFormat::depth32f;

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    image_info.imageType = VK_IMAGE_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    image_info.imageType = VK_IMAGE_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    image_info.imageType = VK_IMAGE_TYPE_3D;
  }
  image_info.extent.width = params.x;
  image_info.extent.height = params.y;
  image_info.extent.depth = params.z;
  image_info.mipLevels = num_mip_levels;
  image_info.arrayLayers = 1;
  image_info.format = buffer_format_ti_to_vk(params.format);
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (params.usage & ImageAllocUsage::Sampled) {
    image_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }

  if (is_depth) {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
  } else {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
  }
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    image_info.queueFamilyIndexCount = 2;
    image_info.pQueueFamilyIndices = queue_family_indices;
  }

  bool export_sharing = params.export_sharing &&
                        this->get_cap(DeviceCapability::vk_has_external_memory);

  VkExternalMemoryImageCreateInfo external_mem_image_create_info = {};
  if (export_sharing) {
    external_mem_image_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external_mem_image_create_info.pNext = NULL;

#ifdef _WIN64
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
    image_info.pNext = &external_mem_image_create_info;
  }

  VmaAllocationCreateInfo alloc_info{};
  if (params.export_sharing) {
  }
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  alloc.image = vkapi::create_image(
      device_, export_sharing ? allocator_export_ : allocator_, &image_info,
      &alloc_info);
  vmaGetAllocationInfo(alloc.image->allocator, alloc.image->allocation,
                       &alloc.alloc_info);

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
  }
  view_info.format = image_info.format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask =
      is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = num_mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  alloc.view = vkapi::create_image_view(device_, alloc.image, &view_info);

  for (int i = 0; i < num_mip_levels; i++) {
    view_info.subresourceRange.baseMipLevel = i;
    view_info.subresourceRange.levelCount = 1;
    alloc.view_lods.push_back(
        vkapi::create_image_view(device_, alloc.image, &view_info));
  }

  if (params.initial_layout != ImageLayout::undefined) {
    image_transition(handle, ImageLayout::undefined, params.initial_layout);
  }

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  TI_TRACE("Allocate VK image {}, alloc_id={}", (void *)alloc.image,
           handle.alloc_id);
#endif

  return handle;
}

void VulkanDevice::destroy_image(DeviceAllocation handle) {
  auto map_pair = image_allocations_.find(handle.alloc_id);

  TI_ASSERT_INFO(map_pair != image_allocations_.end(),
                 "Invalid handle (double free?) {}", handle.alloc_id);

  image_allocations_.erase(handle.alloc_id);
}

vkapi::IVkRenderPass VulkanDevice::get_renderpass(
    const VulkanRenderPassDesc &desc) {
  if (renderpass_pools_.find(desc) != renderpass_pools_.end()) {
    return renderpass_pools_.at(desc);
  }

  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_attachments;

  VkAttachmentReference depth_attachment;

  uint32_t i = 0;
  for (auto [format, clear] : desc.color_attachments) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp =
        clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference &ref = color_attachments.emplace_back();
    ref.attachment = i;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    i += 1;
  }

  if (desc.depth_attachment != VK_FORMAT_UNDEFINED) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = desc.depth_attachment;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    depth_attachment.attachment = i;
    depth_attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }

  VkSubpassDescription subpass{};
  subpass.flags = 0;
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount = 0;
  subpass.pInputAttachments = nullptr;
  subpass.colorAttachmentCount = color_attachments.size();
  subpass.pColorAttachments = color_attachments.data();
  subpass.pResolveAttachments = 0;
  subpass.pDepthStencilAttachment = desc.depth_attachment == VK_FORMAT_UNDEFINED
                                        ? nullptr
                                        : &depth_attachment;
  subpass.preserveAttachmentCount = 0;
  subpass.pPreserveAttachments = nullptr;

  VkRenderPassCreateInfo renderpass_info{};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.pNext = nullptr;
  renderpass_info.flags = 0;
  renderpass_info.attachmentCount = attachments.size();
  renderpass_info.pAttachments = attachments.data();
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 0;
  renderpass_info.pDependencies = nullptr;

  vkapi::IVkRenderPass renderpass =
      vkapi::create_render_pass(device_, &renderpass_info);

  renderpass_pools_.insert({desc, renderpass});

  return renderpass;
}

vkapi::IVkDescriptorSetLayout VulkanDevice::get_desc_set_layout(
    VulkanResourceBinder::Set &set) {
  if (desc_set_layouts_.find(set) == desc_set_layouts_.end()) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (auto &pair : set.bindings) {
      bindings.push_back(VkDescriptorSetLayoutBinding{
          /*binding=*/pair.first, pair.second.type, /*descriptorCount=*/1,
          VK_SHADER_STAGE_ALL,
          /*pImmutableSamplers=*/nullptr});
    }

    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = 0;
    create_info.bindingCount = bindings.size();
    create_info.pBindings = bindings.data();

    auto layout = vkapi::create_descriptor_set_layout(device_, &create_info);
    desc_set_layouts_[set] = layout;

    return layout;
  } else {
    return desc_set_layouts_.at(set);
  }
}

vkapi::IVkDescriptorSet VulkanDevice::alloc_desc_set(
    vkapi::IVkDescriptorSetLayout layout) {
  // TODO: Currently we assume the calling code has called get_desc_set_layout
  // before allocating a desc set. Either we should guard against this or
  // maintain this assumption in other parts of the VulkanBackend
  vkapi::IVkDescriptorSet set =
      vkapi::allocate_descriptor_sets(desc_pool_, layout);

  if (set == nullptr) {
    new_descriptor_pool();
    set = vkapi::allocate_descriptor_sets(desc_pool_, layout);
  }

  return set;
}

void VulkanDevice::create_vma_allocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion =
      this->get_cap(DeviceCapability::vk_api_version);
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;

  VolkDeviceTable table;
  VmaVulkanFunctions vk_vma_functions{0};

  volkLoadDeviceTable(&table, device_);
  vk_vma_functions.vkGetPhysicalDeviceProperties =
      PFN_vkGetPhysicalDeviceProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceProperties"));
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties =
      PFN_vkGetPhysicalDeviceMemoryProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties"));
  vk_vma_functions.vkAllocateMemory = table.vkAllocateMemory;
  vk_vma_functions.vkFreeMemory = table.vkFreeMemory;
  vk_vma_functions.vkMapMemory = table.vkMapMemory;
  vk_vma_functions.vkUnmapMemory = table.vkUnmapMemory;
  vk_vma_functions.vkFlushMappedMemoryRanges = table.vkFlushMappedMemoryRanges;
  vk_vma_functions.vkInvalidateMappedMemoryRanges =
      table.vkInvalidateMappedMemoryRanges;
  vk_vma_functions.vkBindBufferMemory = table.vkBindBufferMemory;
  vk_vma_functions.vkBindImageMemory = table.vkBindImageMemory;
  vk_vma_functions.vkGetBufferMemoryRequirements =
      table.vkGetBufferMemoryRequirements;
  vk_vma_functions.vkGetImageMemoryRequirements =
      table.vkGetImageMemoryRequirements;
  vk_vma_functions.vkCreateBuffer = table.vkCreateBuffer;
  vk_vma_functions.vkDestroyBuffer = table.vkDestroyBuffer;
  vk_vma_functions.vkCreateImage = table.vkCreateImage;
  vk_vma_functions.vkDestroyImage = table.vkDestroyImage;
  vk_vma_functions.vkCmdCopyBuffer = table.vkCmdCopyBuffer;
  vk_vma_functions.vkGetBufferMemoryRequirements2KHR =
      table.vkGetBufferMemoryRequirements2KHR;
  vk_vma_functions.vkGetImageMemoryRequirements2KHR =
      table.vkGetImageMemoryRequirements2KHR;
  vk_vma_functions.vkBindBufferMemory2KHR = table.vkBindBufferMemory2KHR;
  vk_vma_functions.vkBindImageMemory2KHR = table.vkBindImageMemory2KHR;
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties2KHR =
      PFN_vkGetPhysicalDeviceMemoryProperties2KHR(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties2KHR"));
  vk_vma_functions.vkGetDeviceBufferMemoryRequirements =
      table.vkGetDeviceBufferMemoryRequirements;
  vk_vma_functions.vkGetDeviceImageMemoryRequirements =
      table.vkGetDeviceImageMemoryRequirements;

  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  if (get_cap(DeviceCapability::spirv_has_physical_storage_buffer)) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }

  vmaCreateAllocator(&allocatorInfo, &allocator_);

  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);

  std::vector<VkExternalMemoryHandleTypeFlags> flags(
      properties.memoryTypeCount);

  for (int i = 0; i < properties.memoryTypeCount; i++) {
    auto flag = properties.memoryTypes[i].propertyFlags;
    if (flag & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
#ifdef _WIN64
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    } else {
      flags[i] = 0;
    }
  }

  allocatorInfo.pTypeExternalMemoryHandleTypes = flags.data();

  vmaCreateAllocator(&allocatorInfo, &allocator_export_);
}

void VulkanDevice::new_descriptor_pool() {
  std::vector<VkDescriptorPoolSize> pool_sizes{
      {VK_DESCRIPTOR_TYPE_SAMPLER, 64},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 128}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 64;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  desc_pool_ = vkapi::create_descriptor_pool(device_, &pool_info);
}

VkPresentModeKHR choose_swap_present_mode(
    const std::vector<VkPresentModeKHR> &available_present_modes,
    bool vsync,
    bool adaptive) {
  if (vsync) {
    if (adaptive) {
      for (const auto &available_present_mode : available_present_modes) {
        if (available_present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
          return available_present_mode;
        }
      }
    }
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR) {
        return available_present_mode;
      }
    }
  } else {
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return available_present_mode;
      }
    }
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        return available_present_mode;
      }
    }
  }

  if (available_present_modes.size() == 0) {
    throw std::runtime_error("no avialble present modes");
  }

  return available_present_modes[0];
}

VulkanSurface::VulkanSurface(VulkanDevice *device, const SurfaceConfig &config)
    : config_(config), device_(device) {
#ifdef ANDROID
  window_ = (ANativeWindow *)config.window_handle;
#else
  window_ = (GLFWwindow *)config.window_handle;
#endif
  if (window_) {
    if (config.native_surface_handle) {
      surface_ = (VkSurfaceKHR)config.native_surface_handle;
    } else {
#ifdef ANDROID
      VkAndroidSurfaceCreateInfoKHR createInfo{
          .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
          .pNext = nullptr,
          .flags = 0,
          .window = window_};

      vkCreateAndroidSurfaceKHR(device->vk_instance(), &createInfo, nullptr,
                                &surface_);
#else
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      VkResult err = glfwCreateWindowSurface(device->vk_instance(), window_,
                                             NULL, &surface_);
      if (err) {
        TI_ERROR("Failed to create window surface ({})", err);
        return;
      }
#endif
    }

    create_swap_chain();

    image_available_ = vkapi::create_semaphore(device->vk_device(), 0);
  } else {
    ImageParams params = {ImageDimension::d2D,
                          BufferFormat::rgba8,
                          ImageLayout::present_src,
                          config.width,
                          config.height,
                          1,
                          false};
    // screenshot_image_ = device->create_image(params);
    swapchain_images_.push_back(device->create_image(params));
    swapchain_images_.push_back(device->create_image(params));
    width_ = config.width;
    height_ = config.height;
  }
}

void VulkanSurface::create_swap_chain() {
  auto choose_surface_format =
      [](const std::vector<VkSurfaceFormatKHR> &availableFormats) {
        for (const auto &availableFormat : availableFormats) {
          if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
              availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
          }
        }
        return availableFormats[0];
      };

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->vk_physical_device(),
                                            surface_, &capabilities);

  VkBool32 supported = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(device_->vk_physical_device(),
                                       device_->graphics_queue_family_index(),
                                       surface_, &supported);

  if (!supported) {
    TI_ERROR("Selected queue does not support presenting");
    return;
  }

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> surface_formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, surface_formats.data());

  VkSurfaceFormatKHR surface_format = choose_surface_format(surface_formats);

  uint32_t present_mode_count;
  std::vector<VkPresentModeKHR> present_modes;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device_->vk_physical_device(), surface_, &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_->vk_physical_device(),
                                              surface_, &present_mode_count,
                                              present_modes.data());
  }
  VkPresentModeKHR present_mode =
      choose_swap_present_mode(present_modes, config_.vsync, config_.adaptive);

  int width, height;
#ifdef ANDROID
  width = ANativeWindow_getWidth(window_);
  height = ANativeWindow_getHeight(window_);
#else
  glfwGetFramebufferSize(window_, &width, &height);
#endif

  VkExtent2D extent = {uint32_t(width), uint32_t(height)};
  extent.width =
      std::max(capabilities.minImageExtent.width,
               std::min(capabilities.maxImageExtent.width, extent.width));
  extent.height =
      std::max(capabilities.minImageExtent.height,
               std::min(capabilities.maxImageExtent.height, extent.height));
  TI_INFO("Creating suface of {}x{}", width, height);
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  this->width_ = width;
  this->height_ = height;

  VkSwapchainCreateInfoKHR createInfo;
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.surface = surface_;
  createInfo.minImageCount = capabilities.minImageCount;
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = usage;
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.queueFamilyIndexCount = 0;
  createInfo.pQueueFamilyIndices = nullptr;
  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = present_mode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_->vk_device(), &createInfo,
                           kNoVkAllocCallbacks, &swapchain_) != VK_SUCCESS) {
    TI_ERROR("Failed to create swapchain");
    return;
  }

  uint32_t num_images;
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          nullptr);
  std::vector<VkImage> swapchain_images(num_images);
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          swapchain_images.data());

  image_format_ = buffer_format_vk_to_ti(surface_format.format);

  for (VkImage img : swapchain_images) {
    vkapi::IVkImage image = vkapi::create_image(
        device_->vk_device(), img, surface_format.format, VK_IMAGE_TYPE_2D,
        VkExtent3D{uint32_t(width), uint32_t(height), 1}, 1u, 1u, usage);

    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = image->image;
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = image->format;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    vkapi::IVkImageView view =
        vkapi::create_image_view(device_->vk_device(), image, &create_info);

    swapchain_images_.push_back(
        device_->import_vk_image(image, view, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
  }
}

void VulkanSurface::destroy_swap_chain() {
  for (auto alloc : swapchain_images_) {
    std::get<1>(device_->get_vk_image(alloc)) = nullptr;
    device_->destroy_image(alloc);
  }
  swapchain_images_.clear();
  vkDestroySwapchainKHR(device_->vk_device(), swapchain_, nullptr);
}

int VulkanSurface::get_image_count() {
  return swapchain_images_.size();
}

VulkanSurface::~VulkanSurface() {
  if (config_.window_handle) {
    destroy_swap_chain();
    image_available_ = nullptr;
    vkDestroySurfaceKHR(device_->vk_instance(), surface_, nullptr);
  } else {
    for (auto &img : swapchain_images_) {
      device_->destroy_image(img);
    }
    swapchain_images_.clear();
  }
  if (depth_buffer_ != kDeviceNullAllocation) {
    device_->dealloc_memory(depth_buffer_);
  }
  if (screenshot_buffer_ != kDeviceNullAllocation) {
    device_->dealloc_memory(screenshot_buffer_);
  }
}

void VulkanSurface::resize(uint32_t width, uint32_t height) {
  destroy_swap_chain();
  create_swap_chain();
}

std::pair<uint32_t, uint32_t> VulkanSurface::get_size() {
  return std::make_pair(width_, height_);
}

StreamSemaphore VulkanSurface::acquire_next_image() {
  if (!config_.window_handle) {
    image_index_ = (image_index_ + 1) % swapchain_images_.size();
    return nullptr;
  } else {
    vkAcquireNextImageKHR(device_->vk_device(), swapchain_, UINT64_MAX,
                          image_available_->semaphore, VK_NULL_HANDLE,
                          &image_index_);
    return std::make_shared<VulkanStreamSemaphoreObject>(image_available_);
  }
}

DeviceAllocation VulkanSurface::get_target_image() {
  return swapchain_images_[image_index_];
}

BufferFormat VulkanSurface::image_format() {
  return image_format_;
}

void VulkanSurface::present_image(
    const std::vector<StreamSemaphore> &wait_semaphores) {
  std::vector<VkSemaphore> vk_wait_semaphores;

  // Already transitioned to `present_src` at the end of the render pass.
  // device_->image_transition(get_target_image(),
  // ImageLayout::color_attachment,
  //                          ImageLayout::present_src);

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = vk_wait_semaphores.size();
  presentInfo.pWaitSemaphores = vk_wait_semaphores.data();
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain_;
  presentInfo.pImageIndices = &image_index_;
  presentInfo.pResults = nullptr;

  vkQueuePresentKHR(device_->graphics_queue(), &presentInfo);

  device_->wait_idle();
}

DeviceAllocation VulkanSurface::get_depth_data(DeviceAllocation &depth_alloc) {
  auto *stream = device_->get_graphics_stream();

  auto [w, h] = get_size();
  size_t size_bytes = w * h * 4;

  if (depth_buffer_ == kDeviceNullAllocation) {
    Device::AllocParams params{size_bytes, /*host_wrtie*/ false,
                               /*host_read*/ true, /*export_sharing*/ false,
                               AllocUsage::Uniform};
    depth_buffer_ = device_->allocate_memory(params);
  }

  std::unique_ptr<CommandList> cmd_list{nullptr};

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = w;
  copy_params.image_extent.y = h;
  copy_params.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
  cmd_list = stream->new_command_list();
  cmd_list->image_transition(depth_alloc, ImageLayout::depth_attachment,
                             ImageLayout::transfer_src);
  cmd_list->image_to_buffer(depth_buffer_.get_ptr(), depth_alloc,
                            ImageLayout::transfer_src, copy_params);
  cmd_list->image_transition(depth_alloc, ImageLayout::transfer_src,
                             ImageLayout::depth_attachment);
  stream->submit_synced(cmd_list.get());

  return depth_buffer_;
}

DeviceAllocation VulkanSurface::get_image_data() {
  auto *stream = device_->get_graphics_stream();
  DeviceAllocation img_alloc = swapchain_images_[image_index_];
  auto [w, h] = get_size();
  size_t size_bytes = w * h * 4;

  /*
  if (screenshot_image_ == kDeviceNullAllocation) {
    ImageParams params = {ImageDimension::d2D,
                          BufferFormat::rgba8,
                          ImageLayout::transfer_dst,
                          w,
                          h,
                          1,
                          false};
    screenshot_image_ = device_->create_image(params);
  }
  */

  if (screenshot_buffer_ == kDeviceNullAllocation) {
    Device::AllocParams params{size_bytes, /*host_wrtie*/ false,
                               /*host_read*/ true, /*export_sharing*/ false,
                               AllocUsage::Uniform};
    screenshot_buffer_ = device_->allocate_memory(params);
  }

  std::unique_ptr<CommandList> cmd_list{nullptr};

  /*
  if (config_.window_handle) {
    // TODO: check if blit is supported, and use copy_image if not
    cmd_list = stream->new_command_list();
    cmd_list->blit_image(screenshot_image_, img_alloc,
                         ImageLayout::transfer_dst, ImageLayout::transfer_src,
                         {w, h, 1});
    cmd_list->image_transition(screenshot_image_, ImageLayout::transfer_dst,
                               ImageLayout::transfer_src);
    stream->submit_synced(cmd_list.get());
  }
  */

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = w;
  copy_params.image_extent.y = h;
  copy_params.image_aspect_flag = VK_IMAGE_ASPECT_COLOR_BIT;
  cmd_list = stream->new_command_list();
  cmd_list->image_transition(img_alloc, ImageLayout::present_src,
                             ImageLayout::transfer_src);
  // TODO: directly map the image to cpu memory
  cmd_list->image_to_buffer(screenshot_buffer_.get_ptr(), img_alloc,
                            ImageLayout::transfer_src, copy_params);
  cmd_list->image_transition(img_alloc, ImageLayout::transfer_src,
                             ImageLayout::present_src);
  /*
  if (config_.window_handle) {
    cmd_list->image_transition(screenshot_image_, ImageLayout::transfer_src,
                               ImageLayout::transfer_dst);
  }
  */
  stream->submit_synced(cmd_list.get());

  return screenshot_buffer_;
}

VulkanStream::VulkanStream(VulkanDevice &device,
                           VkQueue queue,
                           uint32_t queue_family_index)
    : device_(device), queue_(queue), queue_family_index_(queue_family_index) {
  command_pool_ = vkapi::create_command_pool(
      device_.vk_device(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      queue_family_index);
}

VulkanStream::~VulkanStream() {
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
