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

#include "spirv_reflect.h"

namespace taichi::lang {
namespace vulkan {

using namespace rhi_impl;

const BidirMap<BufferFormat, VkFormat> buffer_format_map = {
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

RhiReturn<VkFormat> buffer_format_ti_to_vk(BufferFormat f) {
  if (!buffer_format_map.exists(f)) {
    RHI_LOG_ERROR("BufferFormat cannot be mapped to vk");
    return {RhiResult::not_supported, VK_FORMAT_UNDEFINED};
  }
  return {RhiResult::success, buffer_format_map.at(f)};
}

RhiReturn<BufferFormat> buffer_format_vk_to_ti(VkFormat f) {
  if (!buffer_format_map.exists(f)) {
    RHI_LOG_ERROR("VkFormat cannot be mapped to ti");
    return {RhiResult::not_supported, BufferFormat::unknown};
  }
  return {RhiResult::success, buffer_format_map.backend2rhi.at(f)};
}

const BidirMap<ImageLayout, VkImageLayout> image_layout_map = {
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
  if (!image_layout_map.exists(layout)) {
    RHI_LOG_ERROR("ImageLayout cannot be mapped to vk");
    return VK_IMAGE_LAYOUT_UNDEFINED;
  }
  return image_layout_map.at(layout);
}

const BidirMap<BlendOp, VkBlendOp> blend_op_map = {
    {BlendOp::add, VK_BLEND_OP_ADD},
    {BlendOp::subtract, VK_BLEND_OP_SUBTRACT},
    {BlendOp::reverse_subtract, VK_BLEND_OP_REVERSE_SUBTRACT},
    {BlendOp::min, VK_BLEND_OP_MIN},
    {BlendOp::max, VK_BLEND_OP_MAX}};

RhiReturn<VkBlendOp> blend_op_ti_to_vk(BlendOp op) {
  if (!blend_op_map.exists(op)) {
    RHI_LOG_ERROR("BlendOp cannot be mapped to vk");
    return {RhiResult::not_supported, VK_BLEND_OP_ADD};
  }
  return {RhiResult::success, blend_op_map.at(op)};
}

const BidirMap<BlendFactor, VkBlendFactor> blend_factor_map = {
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

RhiReturn<VkBlendFactor> blend_factor_ti_to_vk(BlendFactor factor) {
  if (!blend_factor_map.exists(factor)) {
    RHI_LOG_ERROR("BlendFactor cannot be mapped to vk");
    return {RhiResult::not_supported, VK_BLEND_FACTOR_ONE};
  }
  return {RhiResult::success, blend_factor_map.at(factor)};
}

VulkanPipelineCache::VulkanPipelineCache(VulkanDevice *device,
                                         size_t initial_size,
                                         const void *initial_data)
    : device_(device) {
  cache_ = vkapi::create_pipeline_cache(device_->vk_device(), 0, initial_size,
                                        initial_data);
}

VulkanPipelineCache ::~VulkanPipelineCache() {
}

void *VulkanPipelineCache::data() noexcept {
  try {
    data_shadow_.resize(size());
    size_t size = 0;
    vkGetPipelineCacheData(device_->vk_device(), cache_->cache, &size,
                           data_shadow_.data());
  } catch (std::bad_alloc &) {
    return nullptr;
  }

  return data_shadow_.data();
}

size_t VulkanPipelineCache::size() const noexcept {
  size_t size = 0;
  vkGetPipelineCacheData(device_->vk_device(), cache_->cache, &size, nullptr);
  return size;
}

VulkanPipeline::VulkanPipeline(const Params &params)
    : ti_device_(*params.device),
      device_(params.device->vk_device()),
      name_(params.name) {
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
    : ti_device_(*params.device),
      device_(params.device->vk_device()),
      name_(params.name) {
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
  VkResult res = vkCreateShaderModule(device, &create_info, kNoVkAllocCallbacks,
                                      &shader_module);
  RHI_THROW_UNLESS(res == VK_SUCCESS,
                   std::runtime_error("vkCreateShaderModule failed"));
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

vkapi::IVkPipeline VulkanPipeline::graphics_pipeline_dynamic(
    const VulkanRenderPassDesc &renderpass_desc) {
  if (graphics_pipeline_dynamic_.find(renderpass_desc) !=
      graphics_pipeline_dynamic_.end()) {
    return graphics_pipeline_dynamic_.at(renderpass_desc);
  }

  std::vector<VkFormat> color_attachment_formats;
  for (const auto &color_attachment : renderpass_desc.color_attachments) {
    color_attachment_formats.push_back(color_attachment.first);
  }

  VkPipelineRenderingCreateInfoKHR rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  rendering_info.pNext = nullptr;
  rendering_info.viewMask = 0;
  rendering_info.colorAttachmentCount =
      renderpass_desc.color_attachments.size();
  rendering_info.pColorAttachmentFormats = color_attachment_formats.data();
  rendering_info.depthAttachmentFormat = renderpass_desc.depth_attachment;
  rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  vkapi::IVkPipeline pipeline = vkapi::create_graphics_pipeline_dynamic(
      device_, &graphics_pipeline_template_->pipeline_info, &rendering_info,
      pipeline_layout_);

  graphics_pipeline_dynamic_[renderpass_desc] = pipeline;

  return pipeline;
}

void VulkanPipeline::create_descriptor_set_layout(const Params &params) {
  for (auto &code_view : params.code) {
    SpvReflectShaderModule module;
    SpvReflectResult result =
        spvReflectCreateShaderModule(code_view.size, code_view.data, &module);
    RHI_THROW_UNLESS(result == SPV_REFLECT_RESULT_SUCCESS,
                     std::runtime_error("spvReflectCreateShaderModule failed"));

    uint32_t set_count = 0;
    result = spvReflectEnumerateDescriptorSets(&module, &set_count, nullptr);
    RHI_THROW_UNLESS(result == SPV_REFLECT_RESULT_SUCCESS,
                     std::runtime_error("Failed to enumerate number of sets"));
    std::vector<SpvReflectDescriptorSet *> desc_sets(set_count);
    result = spvReflectEnumerateDescriptorSets(&module, &set_count,
                                               desc_sets.data());
    RHI_THROW_UNLESS(
        result == SPV_REFLECT_RESULT_SUCCESS,
        std::runtime_error("spvReflectEnumerateDescriptorSets failed"));

    for (SpvReflectDescriptorSet *desc_set : desc_sets) {
      uint32_t set_index = desc_set->set;
      if (set_templates_.find(set_index) == set_templates_.end()) {
        set_templates_.insert({set_index, VulkanResourceSet(&ti_device_)});
      }
      VulkanResourceSet &set = set_templates_.at(set_index);

      for (int i = 0; i < desc_set->binding_count; i++) {
        SpvReflectDescriptorBinding *desc_binding = desc_set->bindings[i];

        if (desc_binding->descriptor_type ==
            SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          set.rw_buffer(desc_binding->binding, kDeviceNullPtr, 0);
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
          set.buffer(desc_binding->binding, kDeviceNullPtr, 0);
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
          set.image(desc_binding->binding, kDeviceNullAllocation, {});
        } else if (desc_binding->descriptor_type ==
                   SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
          set.rw_image(desc_binding->binding, kDeviceNullAllocation, {});
        } else {
          RHI_LOG_ERROR("Unrecognized binding ignored");
        }
      }
    }

    // Handle special vertex shaders stuff
    // if (code_view.stage == VK_SHADER_STAGE_VERTEX_BIT) {
    //   uint32_t attrib_count;
    //   result =
    //       spvReflectEnumerateInputVariables(&module, &attrib_count, nullptr);
    //   RHI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
    //   std::vector<SpvReflectInterfaceVariable *> attribs(attrib_count);
    //   result = spvReflectEnumerateInputVariables(&module, &attrib_count,
    //                                               attribs.data());
    //   RHI_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);

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
      RHI_THROW_UNLESS(
          result == SPV_REFLECT_RESULT_SUCCESS,
          std::runtime_error("Failed to enumerate number of output vars"));

      std::vector<SpvReflectInterfaceVariable *> variables(render_target_count);
      result = spvReflectEnumerateOutputVariables(&module, &render_target_count,
                                                  variables.data());

      RHI_THROW_UNLESS(
          result == SPV_REFLECT_RESULT_SUCCESS,
          std::runtime_error("spvReflectEnumerateOutputVariables failed"));

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

  // A program can have no binding sets at all.
  if (set_templates_.size()) {
    // We need to verify the set layouts are all continous
    uint32_t max_set = 0;
    for (auto &[index, layout_template] : set_templates_) {
      max_set = std::max(index, max_set);
    }
    RHI_THROW_UNLESS(
        max_set + 1 == set_templates_.size(),
        std::invalid_argument("Sets must be continous & start with 0"));

    set_layouts_.resize(set_templates_.size(), nullptr);
    for (auto &[index, layout_template] : set_templates_) {
      set_layouts_[index] = ti_device_.get_desc_set_layout(layout_template);
    }
  }
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
  std::array<char, 512> msg_buf;
  RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                     "Compiling Vulkan pipeline %s", params.name.data());
  RHI_LOG_DEBUG(msg_buf.data());
  pipeline_ = vkapi::create_compute_pipeline(device_, 0, shader_stages_[0],
                                             pipeline_layout_, params.cache);
}

void VulkanPipeline::create_graphics_pipeline(
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs) {
  // Use dynamic viewport state. These two are just dummies
  VkViewport viewport{};
  viewport.width = 1;
  viewport.height = 1;
  viewport.x = 0;
  viewport.y = 0;
  viewport.minDepth = 0.0;
  viewport.maxDepth = 1.0;

  VkRect2D scissor{/*offset*/ {0, 0}, /*extent*/ {1, 1}};

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
    auto [result, vk_format] = buffer_format_ti_to_vk(attr.format);
    RHI_ASSERT(result == RhiResult::success);
    desc.format = vk_format;
    assert(desc.format != VK_FORMAT_UNDEFINED);
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
    if (raster_params.blending.size() != color_blending.attachmentCount) {
      std::array<char, 256> buf;
      RHI_DEBUG_SNPRINTF(
          buf.data(), buf.size(),
          "RasterParams::blending (size=%u) must either be zero sized "
          "or match the number of fragment shader outputs (size=%u).",
          uint32_t(raster_params.blending.size()),
          uint32_t(color_blending.attachmentCount));
      RHI_LOG_ERROR(buf.data());
      RHI_ASSERT(false);
    }

    for (int i = 0; i < raster_params.blending.size(); i++) {
      auto &state = graphics_pipeline_template_->blend_attachments[i];
      auto &ti_param = raster_params.blending[i];
      state.blendEnable = ti_param.enable;
      if (ti_param.enable) {
        {
          auto [res, op] = blend_op_ti_to_vk(ti_param.color.op);
          RHI_ASSERT(res == RhiResult::success);
          state.colorBlendOp = op;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.color.src_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.srcColorBlendFactor = factor;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.color.dst_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.dstColorBlendFactor = factor;
        }
        {
          auto [res, op] = blend_op_ti_to_vk(ti_param.alpha.op);
          RHI_ASSERT(res == RhiResult::success);
          state.alphaBlendOp = op;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.alpha.src_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.srcAlphaBlendFactor = factor;
        }
        {
          auto [res, factor] = blend_factor_ti_to_vk(ti_param.alpha.dst_factor);
          RHI_ASSERT(res == RhiResult::success);
          state.dstAlphaBlendFactor = factor;
        }
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

VulkanResourceSet::VulkanResourceSet(VulkanDevice *device) : device_(device) {
}

VulkanResourceSet::~VulkanResourceSet() {
}

ShaderResourceSet &VulkanResourceSet::rw_buffer(uint32_t binding,
                                                DevicePtr ptr,
                                                size_t size) {
  dirty_ = true;

  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  bindings_[binding] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        Buffer{buffer, ptr.offset, size}};
  return *this;
}

ShaderResourceSet &VulkanResourceSet::rw_buffer(uint32_t binding,
                                                DeviceAllocation alloc) {
  return rw_buffer(binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

ShaderResourceSet &VulkanResourceSet::buffer(uint32_t binding,
                                             DevicePtr ptr,
                                             size_t size) {
  dirty_ = true;

  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  bindings_[binding] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        Buffer{buffer, ptr.offset, size}};
  return *this;
}

ShaderResourceSet &VulkanResourceSet::buffer(uint32_t binding,
                                             DeviceAllocation alloc) {
  return buffer(binding, alloc.get_ptr(0), VK_WHOLE_SIZE);
}

ShaderResourceSet &VulkanResourceSet::image(uint32_t binding,
                                            DeviceAllocation alloc,
                                            ImageSamplerConfig sampler_config) {
  dirty_ = true;

  vkapi::IVkSampler sampler = nullptr;
  vkapi::IVkImageView view = nullptr;

  if (alloc != kDeviceNullAllocation) {
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

    sampler = vkapi::create_sampler(device_->vk_device(), sampler_info);
    view = device_->get_vk_imageview(alloc);
  }

  bindings_[binding] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        Texture{view, sampler}};

  return *this;
}

ShaderResourceSet &VulkanResourceSet::rw_image(uint32_t binding,
                                               DeviceAllocation alloc,
                                               int lod) {
  dirty_ = true;

  vkapi::IVkImageView view = (alloc != kDeviceNullAllocation)
                                 ? device_->get_vk_lod_imageview(alloc, lod)
                                 : nullptr;

  bindings_[binding] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, Image{view}};

  return *this;
}

RhiReturn<vkapi::IVkDescriptorSet> VulkanResourceSet::finalize() {
  if (!dirty_ && set_) {
    // If nothing changed directly return the set
    return {RhiResult::success, set_};
  }

  if (bindings_.size() <= 0) {
    // A set can't be empty
    return {RhiResult::invalid_usage, nullptr};
  }

  vkapi::IVkDescriptorSetLayout new_layout =
      device_->get_desc_set_layout(*this);
  if (new_layout != layout_) {
    // Layout changed, reset `set`
    set_ = nullptr;
    layout_ = new_layout;
  }

  if (!set_) {
    // If set_ is null, create a new one
    auto [status, new_set] = device_->alloc_desc_set(layout_);
    if (status != RhiResult::success) {
      return {status, nullptr};
    }
    set_ = new_set;
  }

  std::forward_list<VkDescriptorBufferInfo> buffer_infos;
  std::forward_list<VkDescriptorImageInfo> image_infos;
  std::vector<VkWriteDescriptorSet> desc_writes;

  set_->ref_binding_objs.clear();

  for (auto &pair : bindings_) {
    uint32_t binding = pair.first;
    VkDescriptorType type = pair.second.type;
    auto &resource = pair.second.res;

    VkWriteDescriptorSet &write = desc_writes.emplace_back();
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = nullptr;
    write.dstSet = set_->set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = nullptr;
    write.pBufferInfo = nullptr;
    write.pTexelBufferView = nullptr;

    if (Buffer *buf = std::get_if<Buffer>(&resource)) {
      VkDescriptorBufferInfo &buffer_info = buffer_infos.emplace_front();
      buffer_info.buffer = buf->buffer ? buf->buffer->buffer : VK_NULL_HANDLE;
      buffer_info.offset = buf->offset;
      buffer_info.range = buf->size;

      write.pBufferInfo = &buffer_info;
      if (buf->buffer) {
        set_->ref_binding_objs.push_back(buf->buffer);
      }
    } else if (Image *img = std::get_if<Image>(&resource)) {
      VkDescriptorImageInfo &image_info = image_infos.emplace_front();
      image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      image_info.imageView = img->view ? img->view->view : VK_NULL_HANDLE;
      image_info.sampler = VK_NULL_HANDLE;

      write.pImageInfo = &image_info;
      if (img->view) {
        set_->ref_binding_objs.push_back(img->view);
      }
    } else if (Texture *tex = std::get_if<Texture>(&resource)) {
      VkDescriptorImageInfo &image_info = image_infos.emplace_front();
      image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      image_info.imageView = tex->view ? tex->view->view : VK_NULL_HANDLE;
      image_info.sampler =
          tex->sampler ? tex->sampler->sampler : VK_NULL_HANDLE;

      write.pImageInfo = &image_info;
      if (tex->view) {
        set_->ref_binding_objs.push_back(tex->view);
      }
      if (tex->sampler) {
        set_->ref_binding_objs.push_back(tex->sampler);
      }
    } else {
      RHI_LOG_ERROR("Ignoring unsupported Descriptor Type");
    }
  }

  vkUpdateDescriptorSets(device_->vk_device(), desc_writes.size(),
                         desc_writes.data(), /*descriptorCopyCount=*/0,
                         /*pDescriptorCopies=*/nullptr);

  dirty_ = false;

  return {RhiResult::success, set_};
}

RasterResources &VulkanRasterResources::vertex_buffer(DevicePtr ptr,
                                                      uint32_t binding) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  if (buffer == nullptr) {
    vertex_buffers.erase(binding);
  } else {
    vertex_buffers[binding] = {buffer, ptr.offset};
  }
  return *this;
}

RasterResources &VulkanRasterResources::index_buffer(DevicePtr ptr,
                                                     size_t index_width) {
  vkapi::IVkBuffer buffer =
      (ptr != kDeviceNullPtr) ? device_->get_vkbuffer(ptr) : nullptr;
  if (buffer == nullptr) {
    index_binding = BufferBinding();
    index_type = VK_INDEX_TYPE_MAX_ENUM;
  } else {
    index_binding = {buffer, ptr.offset};
    if (index_width == 32) {
      index_type = VK_INDEX_TYPE_UINT32;
    } else if (index_width == 16) {
      index_type = VK_INDEX_TYPE_UINT16;
    }
  }
  return *this;
}

VulkanCommandList::VulkanCommandList(VulkanDevice *ti_device,
                                     VulkanStream *stream,
                                     vkapi::IVkCommandBuffer buffer)
    : ti_device_(ti_device),
      stream_(stream),
      device_(ti_device->vk_device()),
      buffer_(buffer) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  vkBeginCommandBuffer(buffer->buffer, &info);
}

VulkanCommandList::~VulkanCommandList() {
}

void VulkanCommandList::bind_pipeline(Pipeline *p) noexcept {
  auto pipeline = static_cast<VulkanPipeline *>(p);

  if (current_pipeline_ == pipeline)
    return;

  if (pipeline->is_graphics()) {
    vkapi::IVkPipeline vk_pipeline =
        ti_device_->vk_caps().dynamic_rendering
            ? pipeline->graphics_pipeline_dynamic(current_renderpass_desc_)
            : pipeline->graphics_pipeline(current_renderpass_desc_,
                                          current_renderpass_);
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vk_pipeline->pipeline);

    VkViewport viewport{};
    viewport.width = viewport_width_;
    viewport.height = viewport_height_;
    viewport.x = 0;
    viewport.y = 0;
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    VkRect2D scissor{/*offset*/ {0, 0},
                     /*extent*/ {viewport_width_, viewport_height_}};

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

RhiResult VulkanCommandList::bind_shader_resources(ShaderResourceSet *res,
                                                   int set_index) noexcept {
  VulkanResourceSet *set = static_cast<VulkanResourceSet *>(res);
  if (set->get_bindings().size() <= 0) {
    return RhiResult::success;
  }

  auto [status, vk_set] = set->finalize();
  if (status != RhiResult::success) {
    return status;
  }

  vkapi::IVkDescriptorSetLayout set_layout = set->get_layout();

  if (current_pipeline_->pipeline_layout()->ref_desc_layouts.empty() ||
      current_pipeline_->pipeline_layout()->ref_desc_layouts[set_index] !=
          set_layout) {
    // WARN: we have a layout mismatch
    RHI_LOG_ERROR("Layout mismatch");

    auto &templates = current_pipeline_->get_resource_set_templates();
    VulkanResourceSet &set_template = templates.at(set_index);

    for (const auto &template_binding : set_template.get_bindings()) {
      std::array<char, 256> msg_buf;
      RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                         "Template binding %d: (VkDescriptorType) %d",
                         template_binding.first, template_binding.second.type);
      RHI_LOG_ERROR(msg_buf.data());
    }

    for (const auto &binding : set->get_bindings()) {
      std::array<char, 256> msg_buf;
      RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                         "Binding %d: (VkDescriptorType) %d", binding.first,
                         binding.second.type);
      RHI_LOG_ERROR(msg_buf.data());
    }

    return RhiResult::invalid_usage;
  }

  VkPipelineLayout pipeline_layout =
      current_pipeline_->pipeline_layout()->layout;
  VkPipelineBindPoint bind_point = current_pipeline_->is_graphics()
                                       ? VK_PIPELINE_BIND_POINT_GRAPHICS
                                       : VK_PIPELINE_BIND_POINT_COMPUTE;

  vkCmdBindDescriptorSets(buffer_->buffer, bind_point, pipeline_layout,
                          /*firstSet=*/set_index,
                          /*descriptorSetCount=*/1, &vk_set->set,
                          /*dynamicOffsetCount=*/0,
                          /*pDynamicOffsets=*/nullptr);
  buffer_->refs.push_back(vk_set);

  return RhiResult::success;
}

RhiResult VulkanCommandList::bind_raster_resources(
    RasterResources *_res) noexcept {
  VulkanRasterResources *res = static_cast<VulkanRasterResources *>(_res);

  if (!current_pipeline_->is_graphics()) {
    return RhiResult::invalid_usage;
  }

  if (res->index_binding.buffer != nullptr) {
    // We have a valid index buffer
    if (res->index_type >= VK_INDEX_TYPE_MAX_ENUM) {
      return RhiResult::not_supported;
    }

    vkapi::IVkBuffer index_buffer = res->index_binding.buffer;
    vkCmdBindIndexBuffer(buffer_->buffer, index_buffer->buffer,
                         res->index_binding.offset, res->index_type);
    buffer_->refs.push_back(index_buffer);
  }

  for (auto &[binding, buffer] : res->vertex_buffers) {
    VkDeviceSize offset_vk = buffer.offset;
    vkCmdBindVertexBuffers(buffer_->buffer, binding, 1, &buffer.buffer->buffer,
                           &offset_vk);
    buffer_->refs.push_back(buffer.buffer);
  }

  return RhiResult::success;
}

void VulkanCommandList::buffer_barrier(DevicePtr ptr, size_t size) noexcept {
  auto buffer = ti_device_->get_vkbuffer(ptr);
  size_t buffer_size = ti_device_->get_vkbuffer_size(ptr);

  // Clamp to buffer size
  if (ptr.offset > buffer_size) {
    return;
  }

  if (saturate_uadd<size_t>(ptr.offset, size) > buffer_size) {
    size = VK_WHOLE_SIZE;
  }

  VkBufferMemoryBarrier barrier{};
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

void VulkanCommandList::buffer_barrier(DeviceAllocation alloc) noexcept {
  buffer_barrier(DevicePtr{alloc, 0}, std::numeric_limits<size_t>::max());
}

void VulkanCommandList::memory_barrier() noexcept {
  VkMemoryBarrier barrier{};
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

void VulkanCommandList::buffer_copy(DevicePtr dst,
                                    DevicePtr src,
                                    size_t size) noexcept {
  size_t src_size = ti_device_->get_vkbuffer_size(src);
  size_t dst_size = ti_device_->get_vkbuffer_size(dst);

  // Clamp to minimum available size
  if (saturate_uadd<size_t>(src.offset, size) > src_size) {
    size = saturate_usub<size_t>(src_size, src.offset);
  }
  if (saturate_uadd<size_t>(dst.offset, size) > dst_size) {
    size = saturate_usub<size_t>(dst_size, dst.offset);
  }

  if (size == 0) {
    return;
  }

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

void VulkanCommandList::buffer_fill(DevicePtr ptr,
                                    size_t size,
                                    uint32_t data) noexcept {
  // Align to 4 bytes
  ptr.offset = ptr.offset & size_t(-4);

  auto buffer = ti_device_->get_vkbuffer(ptr);
  size_t buffer_size = ti_device_->get_vkbuffer_size(ptr);

  // Check for overflow
  if (ptr.offset > buffer_size) {
    return;
  }

  if (saturate_uadd<size_t>(ptr.offset, size) > buffer_size) {
    size = VK_WHOLE_SIZE;
  }

  vkCmdFillBuffer(buffer_->buffer, buffer->buffer, ptr.offset, size, data);
  buffer_->refs.push_back(buffer);
}

RhiResult VulkanCommandList::dispatch(uint32_t x,
                                      uint32_t y,
                                      uint32_t z) noexcept {
  auto &dev_props = ti_device_->get_vk_physical_device_props();
  if (x > dev_props.limits.maxComputeWorkGroupCount[0] ||
      y > dev_props.limits.maxComputeWorkGroupCount[1] ||
      z > dev_props.limits.maxComputeWorkGroupCount[2]) {
    return RhiResult::not_supported;
  }
  vkCmdDispatch(buffer_->buffer, x, y, z);
  return RhiResult::success;
}

vkapi::IVkCommandBuffer VulkanCommandList::vk_command_buffer() {
  return buffer_;
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

  VkRect2D render_area{/*offset*/ {x0, y0},
                       /*extent*/ {uint32_t(x1 - x0), uint32_t(y1 - y0)}};

  viewport_width_ = render_area.extent.width;
  viewport_height_ = render_area.extent.height;

  // Dynamic rendering codepath
  if (ti_device_->vk_caps().dynamic_rendering) {
    current_dynamic_targets_.clear();

    std::vector<VkRenderingAttachmentInfoKHR> color_attachment_infos(
        num_color_attachments);
    for (uint32_t i = 0; i < num_color_attachments; i++) {
      auto [image, view, format] =
          ti_device_->get_vk_image(color_attachments[i]);
      bool clear = color_clear[i];
      rp_desc.color_attachments.emplace_back(format, clear);

      VkRenderingAttachmentInfoKHR &attachment_info = color_attachment_infos[i];
      attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
      attachment_info.pNext = nullptr;
      attachment_info.imageView = view->view;
      attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
      attachment_info.resolveImageView = VK_NULL_HANDLE;
      attachment_info.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      attachment_info.loadOp =
          clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      if (clear) {
        attachment_info.clearValue.color = {
            {clear_colors[i][0], clear_colors[i][1], clear_colors[i][2],
             clear_colors[i][3]}};
      }

      current_dynamic_targets_.push_back(image);
    }

    VkRenderingInfoKHR render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    render_info.pNext = nullptr;
    render_info.flags = 0;
    render_info.renderArea = render_area;
    render_info.layerCount = 1;
    render_info.viewMask = 0;
    render_info.colorAttachmentCount = num_color_attachments;
    render_info.pColorAttachments = color_attachment_infos.data();
    render_info.pDepthAttachment = nullptr;
    render_info.pStencilAttachment = nullptr;

    VkRenderingAttachmentInfo depth_attachment_info{};
    if (depth_attachment) {
      auto [image, view, format] = ti_device_->get_vk_image(*depth_attachment);
      rp_desc.depth_attachment = format;

      depth_attachment_info.sType =
          VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
      depth_attachment_info.pNext = nullptr;
      depth_attachment_info.imageView = view->view;
      depth_attachment_info.imageLayout =
          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      depth_attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
      depth_attachment_info.resolveImageView = VK_NULL_HANDLE;
      depth_attachment_info.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depth_attachment_info.loadOp = depth_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                 : VK_ATTACHMENT_LOAD_OP_LOAD;
      depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depth_attachment_info.clearValue.depthStencil = {0.0, 0};

      render_info.pDepthAttachment = &depth_attachment_info;

      current_dynamic_targets_.push_back(image);
    } else {
      rp_desc.depth_attachment = VK_FORMAT_UNDEFINED;
    }

    vkCmdBeginRenderingKHR(buffer_->buffer, &render_info);

    return;
  }

  // VkRenderpass & VkFramebuffer codepath
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

  current_framebuffer_ = ti_device_->get_framebuffer(fb_desc);

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
  if (ti_device_->vk_caps().dynamic_rendering) {
    vkCmdEndRenderingKHR(buffer_->buffer);

    if (0) {
      std::vector<VkImageMemoryBarrier> memory_barriers(
          current_dynamic_targets_.size());
      for (int i = 0; i < current_dynamic_targets_.size(); i++) {
        VkImageMemoryBarrier &barrier = memory_barriers[i];
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // FIXME: Change this spec to stay in color attachment
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = current_dynamic_targets_[i]->image;
        barrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
      }

      vkCmdPipelineBarrier(buffer_->buffer,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           /*dependencyFlags=*/0, /*memoryBarrierCount=*/0,
                           /*pMemoryBarriers=*/nullptr,
                           /*bufferMemoryBarrierCount=*/0,
                           /*pBufferMemoryBarriers=*/nullptr,
                           /*imageMemoryBarrierCount=*/memory_barriers.size(),
                           /*pImageMemoryBarriers=*/memory_barriers.data());
    }
    current_dynamic_targets_.clear();

    return;
  }

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
  VkOffset3D blit_size{/*x*/ int(params.width),
                       /*y*/ int(params.height),
                       /*z*/ int(params.depth)};
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

void VulkanCommandList::set_line_width(float width) {
  if (ti_device_->vk_caps().wide_line) {
    vkCmdSetLineWidth(buffer_->buffer, width);
  }
}

vkapi::IVkRenderPass VulkanCommandList::current_renderpass() {
  if (ti_device_->vk_caps().dynamic_rendering) {
    vkapi::IVkRenderPass rp =
        ti_device_->get_renderpass(current_renderpass_desc_);
    buffer_->refs.push_back(rp);
    return rp;
  }
  return current_renderpass_;
}

vkapi::IVkCommandBuffer VulkanCommandList::finalize() {
  if (!finalized_) {
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
  DeviceCapabilityConfig caps{};
  caps.set(DeviceCapability::spirv_version, 0x10000);
  set_caps(std::move(caps));
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
  RHI_ASSERT(new_descriptor_pool() == RhiResult::success &&
             "Failed to allocate initial descriptor pool");

  vkGetPhysicalDeviceProperties(physical_device_, &vk_device_properties_);
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
  vkDeviceWaitIdle(device_);

  allocations_.clear();
  image_allocations_.clear();

  compute_streams_.reset();
  graphics_streams_.reset();

  renderpass_pools_.clear();
  desc_set_layouts_.clear();
  desc_pool_ = nullptr;

  vmaDestroyAllocator(allocator_);
  vmaDestroyAllocator(allocator_export_);
}

RhiResult VulkanDevice::create_pipeline_cache(
    PipelineCache **out_cache,
    size_t initial_size,
    const void *initial_data) noexcept {
  try {
    *out_cache = new VulkanPipelineCache(this, initial_size, initial_data);
  } catch (std::bad_alloc &) {
    *out_cache = nullptr;
    return RhiResult::out_of_memory;
  }
  return RhiResult::success;
}

RhiResult VulkanDevice::create_pipeline(Pipeline **out_pipeline,
                                        const PipelineSourceDesc &src,
                                        std::string name,
                                        PipelineCache *cache) noexcept {
  if (src.type != PipelineSourceType::spirv_binary ||
      src.stage != PipelineStageType::compute) {
    return RhiResult::invalid_usage;
  }

  if (src.data == nullptr || src.size == 0) {
    RHI_LOG_ERROR("pipeline source cannot be empty");
    return RhiResult::invalid_usage;
  }

  SpirvCodeView code;
  code.data = (uint32_t *)src.data;
  code.size = src.size;
  code.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  VulkanPipeline::Params params;
  params.code = {code};
  params.device = this;
  params.name = name;
  params.cache =
      cache ? static_cast<VulkanPipelineCache *>(cache)->vk_pipeline_cache()
            : nullptr;

  try {
    *out_pipeline = new VulkanPipeline(params);
  } catch (std::invalid_argument &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::invalid_usage;
  } catch (std::runtime_error &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::error;
  } catch (std::bad_alloc &e) {
    *out_pipeline = nullptr;
    RHI_LOG_ERROR(e.what());
    return RhiResult::out_of_memory;
  }

  return RhiResult::success;
}

RhiResult VulkanDevice::allocate_memory(const AllocParams &params,
                                        DeviceAllocation *out_devalloc) {
  AllocationInternal &alloc = allocations_.acquire();

  RHI_ASSERT(params.size > 0);

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
  external_mem_buffer_create_info.pNext = nullptr;

#ifdef _WIN64
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  bool export_sharing = params.export_sharing && vk_caps().external_memory;

  VmaAllocationCreateInfo alloc_info{};
  if (export_sharing) {
    alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
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
    if (int(params.usage & AllocUsage::Upload)) {
      alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else {
      alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
  } else {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) &&
      ((alloc_info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
       (alloc_info.usage &
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) ||
       (alloc_info.usage &
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) ||
       (alloc_info.usage & VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR))) {
    buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;
  }

  alloc.buffer = vkapi::create_buffer(
      device_, export_sharing ? allocator_export_ : allocator_, &buffer_info,
      &alloc_info);
  if (alloc.buffer == nullptr) {
    return RhiResult::out_of_memory;
  }

  vmaGetAllocationInfo(alloc.buffer->allocator, alloc.buffer->allocation,
                       &alloc.alloc_info);

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = alloc.buffer->buffer;
    info.pNext = nullptr;
    alloc.addr = vkGetBufferDeviceAddressKHR(device_, &info);
  }

  *out_devalloc = DeviceAllocation{this, (uint64_t)&alloc};
  return RhiResult::success;
}

RhiResult VulkanDevice::map_internal(AllocationInternal &alloc_int,
                                     size_t offset,
                                     size_t size,
                                     void **mapped_ptr) {
  if (alloc_int.mapped != nullptr) {
    RHI_LOG_ERROR("Memory can not be mapped multiple times");
    return RhiResult::invalid_usage;
  }

  if (size != VK_WHOLE_SIZE && alloc_int.alloc_info.size < offset + size) {
    RHI_LOG_ERROR("Mapping out of range");
    return RhiResult::invalid_usage;
  }

  VkResult res;
  if (alloc_int.buffer->allocator) {
    res = vmaMapMemory(alloc_int.buffer->allocator,
                       alloc_int.buffer->allocation, &alloc_int.mapped);
    alloc_int.mapped = (uint8_t *)(alloc_int.mapped) + offset;
  } else {
    res = vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                      alloc_int.alloc_info.offset + offset, size, 0,
                      &alloc_int.mapped);
  }

  if (alloc_int.mapped == nullptr || res == VK_ERROR_MEMORY_MAP_FAILED) {
    RHI_LOG_ERROR(
        "cannot map memory, potentially because the memory is not "
        "accessible from the host: ensure your memory is allocated with "
        "`host_read=true` or `host_write=true` (or `host_access=true` in C++ "
        "wrapper)");
    return RhiResult::invalid_usage;
  } else if (res != VK_SUCCESS) {
    std::array<char, 256> msg_buf;
    RHI_DEBUG_SNPRINTF(
        msg_buf.data(), msg_buf.size(),
        "failed to map memory for unknown reasons. VkResult = %d", res);
    RHI_LOG_ERROR(msg_buf.data());
    return RhiResult::error;
  }

  *mapped_ptr = alloc_int.mapped;

  return RhiResult::success;
}

void VulkanDevice::dealloc_memory(DeviceAllocation handle) {
  allocations_.release(&get_alloc_internal(handle));
}

ShaderResourceSet *VulkanDevice::create_resource_set() {
  return new VulkanResourceSet(this);
}

RasterResources *VulkanDevice::create_raster_resources() {
  return new VulkanRasterResources(this);
}

uint64_t VulkanDevice::get_memory_physical_pointer(DeviceAllocation handle) {
  return uint64_t(get_alloc_internal(handle).addr);
}

RhiResult VulkanDevice::map_range(DevicePtr ptr,
                                  uint64_t size,
                                  void **mapped_ptr) {
  return map_internal(get_alloc_internal(ptr), ptr.offset, size, mapped_ptr);
}

RhiResult VulkanDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  return map_internal(get_alloc_internal(alloc), 0, VK_WHOLE_SIZE, mapped_ptr);
}

void VulkanDevice::unmap(DevicePtr ptr) {
  return this->VulkanDevice::unmap(DeviceAllocation(ptr));
}

void VulkanDevice::unmap(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = get_alloc_internal(alloc);

  if (alloc_int.mapped == nullptr) {
    RHI_LOG_ERROR("Unmapping memory that is not mapped");
    return;
  }

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
  auto [cmd, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
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

void VulkanCommandList::begin_profiler_scope(const std::string &kernel_name) {
  auto pool = vkapi::create_query_pool(ti_device_->vk_device());
  vkCmdResetQueryPool(buffer_->buffer, pool->query_pool, 0, 2);
  vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      pool->query_pool, 0);
  ti_device_->profiler_add_sampler(kernel_name, pool);
}

void VulkanCommandList::end_profiler_scope() {
  auto pool = ti_device_->profiler_get_last_query_pool();
  vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      pool->query_pool, 1);
}

void VulkanDevice::profiler_sync() {
  for (auto &sampler : samplers_) {
    auto kernel_name = sampler.first;
    auto query_pool = sampler.second->query_pool;

    double duration_ms = 0.0;

    uint64_t t[2];
    vkGetQueryPoolResults(vk_device(), query_pool, 0, 2, sizeof(uint64_t) * 2,
                          &t, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    duration_ms = (t[1] - t[0]) * vk_device_properties_.limits.timestampPeriod /
                  1000000.0;
    sampled_records_.push_back(std::make_pair(kernel_name, duration_ms));
  }
  samplers_.clear();
}

std::vector<std::pair<std::string, double>>
VulkanDevice::profiler_flush_sampled_time() {
  auto records_ = sampled_records_;
  sampled_records_.clear();
  return records_;
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

RhiResult VulkanStream::new_command_list(CommandList **out_cmdlist) noexcept {
  vkapi::IVkCommandBuffer buffer =
      vkapi::allocate_command_buffer(command_pool_);

  if (buffer == nullptr) {
    return RhiResult::out_of_memory;
  }

  *out_cmdlist = new VulkanCommandList(&device_, this, buffer);
  return RhiResult::success;
}

StreamSemaphore VulkanStream::submit(
    CommandList *cmdlist_,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  VulkanCommandList *cmdlist = static_cast<VulkanCommandList *>(cmdlist_);
  vkapi::IVkCommandBuffer buffer = cmdlist->finalize();

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
  submitted_cmdbuffers_.push_back(TrackedCmdbuf{fence, buffer});

  BAIL_ON_VK_BAD_RESULT_NO_RETURN(
      vkQueueSubmit(queue_, /*submitCount=*/1, &submit_info,
                    /*fence=*/fence->fence),
      "Vulkan device might be lost (vkQueueSubmit failed)");

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

  device_.profiler_sync();

  submitted_cmdbuffers_.clear();
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

  for (auto &src_desc : src) {
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
  auto &buffer_alloc = get_alloc_internal(alloc);
  return std::make_tuple(buffer_alloc.alloc_info.deviceMemory,
                         buffer_alloc.alloc_info.offset,
                         buffer_alloc.alloc_info.size);
}

vkapi::IVkBuffer VulkanDevice::get_vkbuffer(
    const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = get_alloc_internal(alloc);

  return alloc_int.buffer;
}

size_t VulkanDevice::get_vkbuffer_size(const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = get_alloc_internal(alloc);

  return alloc_int.alloc_info.size;
}

std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat>
VulkanDevice::get_vk_image(const DeviceAllocation &alloc) const {
  const ImageAllocInternal &alloc_int = get_image_alloc_internal(alloc);

  return std::make_tuple(alloc_int.image, alloc_int.view,
                         alloc_int.image->format);
}

vkapi::IVkFramebuffer VulkanDevice::get_framebuffer(
    const VulkanFramebufferDesc &desc) {
  // We won't pool framebuffer and resuse it, as doing so requires hashing the
  // referenced IVkImageView objects, which might destruct unless we hold strong
  // references. Thus doing so is way too ugly, and Vulkan is moving towards
  // dynamic rendering anyways.
  vkapi::IVkFramebuffer framebuffer = vkapi::create_framebuffer(
      0, desc.renderpass, desc.attachments, desc.width, desc.height, 1);

  return framebuffer;
}

DeviceAllocation VulkanDevice::import_vkbuffer(vkapi::IVkBuffer buffer,
                                               size_t size,
                                               VkDeviceMemory memory,
                                               VkDeviceSize offset) {
  AllocationInternal &alloc_int = allocations_.acquire();

  alloc_int.external = true;
  alloc_int.buffer = buffer;
  alloc_int.mapped = nullptr;
  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer->buffer;
    info.pNext = nullptr;
    alloc_int.addr = vkGetBufferDeviceAddress(device_, &info);
  }

  alloc_int.alloc_info.size = size;
  alloc_int.alloc_info.deviceMemory = memory;
  alloc_int.alloc_info.offset = offset;

  return DeviceAllocation{this, reinterpret_cast<uint64_t>(&alloc_int)};
}

DeviceAllocation VulkanDevice::import_vk_image(vkapi::IVkImage image,
                                               vkapi::IVkImageView view,
                                               VkImageLayout layout) {
  ImageAllocInternal &alloc_int = image_allocations_.acquire();

  alloc_int.external = true;
  alloc_int.image = image;
  alloc_int.view = view;
  alloc_int.view_lods.emplace_back(view);

  return DeviceAllocation{this, reinterpret_cast<uint64_t>(&alloc_int)};
}

vkapi::IVkImageView VulkanDevice::get_vk_imageview(
    const DeviceAllocation &alloc) const {
  return std::get<1>(get_vk_image(alloc));
}

vkapi::IVkImageView VulkanDevice::get_vk_lod_imageview(
    const DeviceAllocation &alloc,
    int lod) const {
  return get_image_alloc_internal(alloc).view_lods[lod];
}

DeviceAllocation VulkanDevice::create_image(const ImageParams &params) {
  ImageAllocInternal &alloc = image_allocations_.acquire();

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
  auto [result, vk_format] = buffer_format_ti_to_vk(params.format);
  assert(result == RhiResult::success);
  image_info.format = vk_format;
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

  bool export_sharing = params.export_sharing && vk_caps_.external_memory;

  VkExternalMemoryImageCreateInfo external_mem_image_create_info = {};
  if (export_sharing) {
    external_mem_image_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external_mem_image_create_info.pNext = nullptr;

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
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
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

  DeviceAllocation handle{this, reinterpret_cast<uint64_t>(&alloc)};

  if (params.initial_layout != ImageLayout::undefined) {
    image_transition(handle, ImageLayout::undefined, params.initial_layout);
  }

  return handle;
}

void VulkanDevice::destroy_image(DeviceAllocation handle) {
  image_allocations_.release(&get_image_alloc_internal(handle));
}

vkapi::IVkRenderPass VulkanDevice::get_renderpass(
    const VulkanRenderPassDesc &desc) {
  if (renderpass_pools_.find(desc) != renderpass_pools_.end()) {
    return renderpass_pools_.at(desc);
  }

  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_attachments;

  VkAttachmentReference depth_attachment{};

  uint32_t i = 0;
  for (auto &[format, clear] : desc.color_attachments) {
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
  subpass.pResolveAttachments = nullptr;
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
    VulkanResourceSet &set) {
  if (desc_set_layouts_.find(set) == desc_set_layouts_.end()) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto &pair : set.get_bindings()) {
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

RhiReturn<vkapi::IVkDescriptorSet> VulkanDevice::alloc_desc_set(
    vkapi::IVkDescriptorSetLayout layout) {
  // This returns nullptr if can't allocate (OOM or pool is full)
  vkapi::IVkDescriptorSet set =
      vkapi::allocate_descriptor_sets(desc_pool_, layout);

  if (set == nullptr) {
    RhiResult status = new_descriptor_pool();
    // Allocating new descriptor pool failed
    if (status != RhiResult::success) {
      return {status, nullptr};
    }
    set = vkapi::allocate_descriptor_sets(desc_pool_, layout);
  }

  return {RhiResult::success, set};
}

void VulkanDevice::create_vma_allocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = vk_caps().vk_api_version;
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;

  VolkDeviceTable table;
  VmaVulkanFunctions vk_vma_functions{nullptr};

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
      (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)(std::max(
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2KHR"),
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2")));
  vk_vma_functions.vkGetDeviceBufferMemoryRequirements =
      table.vkGetDeviceBufferMemoryRequirements;
  vk_vma_functions.vkGetDeviceImageMemoryRequirements =
      table.vkGetDeviceImageMemoryRequirements;

  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  if (get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
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

RhiResult VulkanDevice::new_descriptor_pool() {
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
  auto new_desc_pool = vkapi::create_descriptor_pool(device_, &pool_info);

  if (!new_desc_pool) {
    return RhiResult::out_of_memory;
  }

  desc_pool_ = new_desc_pool;

  return RhiResult::success;
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
  width_ = config.width;
  height_ = config.height;

  if (config.native_surface_handle) {
    surface_ = (VkSurfaceKHR)config.native_surface_handle;

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
  if (capabilities.maxImageCount == 0) {
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceCapabilitiesKHR.html
    // When maxImageCount is 0, there is no limit on the number of images.
    capabilities.maxImageCount =
        std::max<uint32_t>(capabilities.minImageCount, 3);
  }

  VkBool32 supported = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(device_->vk_physical_device(),
                                       device_->graphics_queue_family_index(),
                                       surface_, &supported);

  if (!supported) {
    RHI_LOG_ERROR("Selected queue does not support presenting");
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

  VkExtent2D extent = {uint32_t(width_), uint32_t(height_)};
  extent.width =
      std::max(capabilities.minImageExtent.width,
               std::min(capabilities.maxImageExtent.width, extent.width));
  extent.height =
      std::max(capabilities.minImageExtent.height,
               std::min(capabilities.maxImageExtent.height, extent.height));
  {
    std::array<char, 512> msg_buf;
    RHI_DEBUG_SNPRINTF(msg_buf.data(), msg_buf.size(),
                       "Creating suface of %u x %u, present mode %d",
                       extent.width, extent.height, present_mode);
    RHI_LOG_DEBUG(msg_buf.data());
  }
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  this->width_ = extent.width;
  this->height_ = extent.height;

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.surface = surface_;
  createInfo.minImageCount = std::min<uint32_t>(capabilities.maxImageCount, 3);
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
    RHI_LOG_ERROR("Failed to create swapchain");
    return;
  }

  uint32_t num_images;
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          nullptr);
  std::vector<VkImage> swapchain_images(num_images);
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          swapchain_images.data());

  auto [result, image_format] = buffer_format_vk_to_ti(surface_format.format);
  RHI_ASSERT(result == RhiResult::success);
  image_format_ = image_format;

  for (VkImage img : swapchain_images) {
    vkapi::IVkImage image = vkapi::create_image(
        device_->vk_device(), img, surface_format.format, VK_IMAGE_TYPE_2D,
        VkExtent3D{uint32_t(width_), uint32_t(height_), 1}, 1u, 1u, usage);

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
  for (auto &alloc : swapchain_images_) {
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
  if (config_.native_surface_handle) {
    destroy_swap_chain();
    image_available_ = nullptr;
  } else {
    for (auto &img : swapchain_images_) {
      device_->destroy_image(img);
    }
    swapchain_images_.clear();
  }
}

void VulkanSurface::resize(uint32_t width, uint32_t height) {
  destroy_swap_chain();
  this->width_ = width;
  this->height_ = height;
  create_swap_chain();
}

std::pair<uint32_t, uint32_t> VulkanSurface::get_size() {
  return std::make_pair(width_, height_);
}

StreamSemaphore VulkanSurface::acquire_next_image() {
  if (!config_.native_surface_handle) {
    image_index_ = (image_index_ + 1) % uint32_t(swapchain_images_.size());
    return nullptr;
  } else {
    VkResult res = vkAcquireNextImageKHR(
        device_->vk_device(), swapchain_, uint64_t(4 * 1e9),
        image_available_->semaphore, VK_NULL_HANDLE, &image_index_);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
      BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "vkAcquireNextImageKHR failed");
    }
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
}  // namespace taichi::lang
