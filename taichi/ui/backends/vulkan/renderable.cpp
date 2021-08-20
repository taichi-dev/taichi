#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/utils/utils.h"

#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"
#include "taichi/ui/backends/vulkan/renderables/kernels.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

void Renderable::init(const RenderableConfig &config,
                      class Renderer *renderer) {
  config_ = config;
  renderer_ = renderer;
  app_context_ = &renderer->app_context();
}

void Renderable::init_render_resources() {
  create_descriptor_pool();
  create_descriptor_set_layout();
  create_graphics_pipeline();

  create_vertex_buffer();
  create_index_buffer();
  create_uniform_buffers();
  create_storage_buffers();
  create_descriptor_sets();

  if (app_context_->config.ti_arch == Arch::cuda) {
    vertex_buffer_device_ptr_ = (Vertex *)get_memory_pointer(
        vertex_buffer_memory_, config_.vertices_count * sizeof(Vertex),
        app_context_->device());
    index_buffer_device_ptr_ = (int *)get_memory_pointer(
        index_buffer_memory_, config_.indices_count * sizeof(int),
        app_context_->device());
  }
}

void Renderable::update_data(const RenderableInfo &info) {
  int num_vertices = info.vertices.shape[0];
  int num_indices;
  if (info.indices.valid) {
    num_indices = info.indices.shape[0];
    if (info.indices.dtype != PrimitiveType::i32 &&
        info.indices.dtype != PrimitiveType::u32) {
      throw std::runtime_error("dtype needs to be 32-bit ints for indices");
    }
  } else {
    num_indices = num_vertices;
  }
  if (num_vertices > config_.vertices_count ||
      num_indices > config_.indices_count) {
    cleanup_swap_chain();
    cleanup();
    config_.vertices_count = num_vertices;
    config_.indices_count = num_indices;
    init_render_resources();
  }

  if (info.vertices.dtype != PrimitiveType::f32) {
    throw std::runtime_error("dtype needs to be f32 for vertices");
  }

  int num_components = info.vertices.matrix_rows;

  if (info.vertices.field_source == FieldSource::TaichiCuda) {
    update_renderables_vertices_cuda(vertex_buffer_device_ptr_,
                                     (float *)info.vertices.data, num_vertices,
                                     num_components);

    if (info.per_vertex_color.valid) {
      if (info.per_vertex_color.shape[0] != num_vertices) {
        throw std::runtime_error(
            "shape of per_vertex_color should be the same as vertices");
      }
      update_renderables_colors_cuda(vertex_buffer_device_ptr_,
                                     (float *)info.per_vertex_color.data,
                                     num_vertices);
    }

    if (info.normals.valid) {
      if (info.normals.shape[0] != num_vertices) {
        throw std::runtime_error(
            "shape of normals should be the same as vertices");
      }
      update_renderables_normals_cuda(vertex_buffer_device_ptr_,
                                      (float *)info.normals.data, num_vertices);
    }

    if (info.indices.valid) {
      indexed_ = true;
      update_renderables_indices_cuda(index_buffer_device_ptr_,
                                      (int *)info.indices.data, num_indices);
    } else {
      indexed_ = false;
    }

  } else if (info.vertices.field_source == FieldSource::TaichiX64) {
    {
      MappedMemory mapped_vbo(app_context_->device(),
                              staging_vertex_buffer_memory_,
                              config_.vertices_count * sizeof(Vertex));
      MappedMemory mapped_ibo(app_context_->device(),
                              staging_index_buffer_memory_,
                              config_.indices_count * sizeof(int));

      update_renderables_vertices_x64((Vertex *)mapped_vbo.data,
                                      (float *)info.vertices.data, num_vertices,
                                      num_components);
      if (info.per_vertex_color.valid) {
        if (info.per_vertex_color.shape[0] != num_vertices) {
          throw std::runtime_error(
              "shape of per_vertex_color should be the same as vertices");
        }
        update_renderables_colors_x64((Vertex *)mapped_vbo.data,
                                      (float *)info.per_vertex_color.data,
                                      num_vertices);
      }
      if (info.normals.valid) {
        if (info.normals.shape[0] != num_vertices) {
          throw std::runtime_error(
              "shape of normals should be the same as vertices");
        }
        update_renderables_normals_x64((Vertex *)mapped_vbo.data,
                                       (float *)info.normals.data,
                                       num_vertices);
      }
      if (info.indices.valid) {
        indexed_ = true;
        update_renderables_indices_x64((int *)mapped_ibo.data,
                                       (int *)info.indices.data, num_indices);
      } else {
        indexed_ = false;
      }
    }

    copy_buffer(staging_vertex_buffer_, vertex_buffer_,
                config_.vertices_count * sizeof(Vertex),
                app_context_->command_pool(), app_context_->device(),
                app_context_->graphics_queue());
    copy_buffer(staging_index_buffer_, index_buffer_,
                config_.indices_count * sizeof(int),
                app_context_->command_pool(), app_context_->device(),
                app_context_->graphics_queue());
  } else {
    throw std::runtime_error("unsupported field source");
  }
}

void Renderable::create_descriptor_pool() {
  int swap_chain_size = 1;
  
  std::array<VkDescriptorPoolSize, 3> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[0].descriptorCount = static_cast<uint32_t>(swap_chain_size);
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sizes[1].descriptorCount = static_cast<uint32_t>(swap_chain_size);
  pool_sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_sizes[2].descriptorCount = static_cast<uint32_t>(swap_chain_size);

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = static_cast<uint32_t>(swap_chain_size);

  if (vkCreateDescriptorPool(app_context_->device(), &pool_info, nullptr,
                             &descriptor_pool_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void Renderable::create_graphics_pipeline() {
  auto vert_code = read_file(config_.vertex_shader_path);
  auto frag_code = read_file(config_.fragment_shader_path);

  VkShaderModule vert_shader_module =
      create_shader_module(vert_code, app_context_->device());
  VkShaderModule frag_shader_module =
      create_shader_module(frag_code, app_context_->device());

  VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
  vert_shader_stage_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_shader_stage_info.module = vert_shader_module;
  vert_shader_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
  frag_shader_stage_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_shader_stage_info.module = frag_shader_module;
  frag_shader_stage_info.pName = "main";

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {
      vert_shader_stage_info, frag_shader_stage_info};

  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  auto binding_description = Vertex::get_binding_description();
  auto attribute_descriptions = Vertex::get_attribute_descriptions();

  vertex_input_info.vertexBindingDescriptionCount = 1;
  vertex_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attribute_descriptions.size());
  vertex_input_info.pVertexBindingDescriptions = &binding_description;
  vertex_input_info.pVertexAttributeDescriptions =
      attribute_descriptions.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  if (config_.topology_type == TopologyType::Triangles) {
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  } else if (config_.topology_type == TopologyType::Lines) {
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  } else if (config_.topology_type == TopologyType::Points) {
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  } else {
    throw std::runtime_error("invalid topology");
  }

  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = renderer_->swap_chain().swap_chain_extent().width;
  viewport.height = renderer_->swap_chain().swap_chain_extent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {(uint32_t)viewport.width, (uint32_t)viewport.height};

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  VkPipelineLayoutCreateInfo pipeline_layout__info{};
  pipeline_layout__info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout__info.setLayoutCount = 1;
  pipeline_layout__info.pSetLayouts = &descriptor_set_layout_;

  if (vkCreatePipelineLayout(app_context_->device(), &pipeline_layout__info,
                             nullptr, &pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  std::vector<VkDynamicState> dynamic_state_enables = {
      VK_DYNAMIC_STATE_LINE_WIDTH};
  VkPipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.pNext = NULL;
  dynamic_state.pDynamicStates = dynamic_state_enables.data();
  dynamic_state.dynamicStateCount = dynamic_state_enables.size();

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = shader_stages.size();
  pipeline_info.pStages = shader_stages.data();
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.layout = pipeline_layout_;
  pipeline_info.renderPass = renderer_->render_passes()[0];

  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(app_context_->device(), VK_NULL_HANDLE, 1,
                                &pipeline_info, nullptr,
                                &graphics_pipeline_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(app_context_->device(), frag_shader_module, nullptr);
  vkDestroyShaderModule(app_context_->device(), vert_shader_module, nullptr);
}

void Renderable::create_vertex_buffer() {
  VkDeviceSize buffer_size = sizeof(Vertex) * config_.vertices_count;
  create_buffer(
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer_,
      vertex_buffer_memory_, app_context_->device(),
      app_context_->physical_device());

  create_buffer(
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      staging_vertex_buffer_, staging_vertex_buffer_memory_,
      app_context_->device(), app_context_->physical_device());
}

void Renderable::create_index_buffer() {
  VkDeviceSize buffer_size = sizeof(int) * config_.indices_count;
  create_buffer(
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer_, index_buffer_memory_,
      app_context_->device(), app_context_->physical_device());

  create_buffer(
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      staging_index_buffer_, staging_index_buffer_memory_,
      app_context_->device(), app_context_->physical_device());
}

void Renderable::create_uniform_buffers() {
  VkDeviceSize buffer_size = config_.ubo_size;
  if (buffer_size == 0) {
    return;
  }


  create_buffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniform_buffer_, uniform_buffer_memory_,
                app_context_->device(), app_context_->physical_device());
  
}

void Renderable::create_storage_buffers() {
  VkDeviceSize buffer_size = config_.ssbo_size;
  if (buffer_size == 0) {
    return;
  } 
    create_buffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  storage_buffer_, storage_buffer_memory_,
                  app_context_->device(), app_context_->physical_device());
  
}

void Renderable::recreate_swap_chain() {
  create_graphics_pipeline();
  create_uniform_buffers();
  create_storage_buffers();
  create_descriptor_pool();
  create_descriptor_sets();
}

void Renderable::destroy_uniform_buffers() {
  if (config_.ubo_size == 0) {
    return;
  }
   vkDestroyBuffer(app_context_->device(), uniform_buffer_, nullptr);
  vkFreeMemory(app_context_->device(), uniform_buffer_memory_, nullptr);
  
}

void Renderable::destroy_storage_buffers() {
  if (config_.ssbo_size == 0) {
    return;
  } 
    vkDestroyBuffer(app_context_->device(), storage_buffer_, nullptr);
    vkFreeMemory(app_context_->device(), storage_buffer_memory_, nullptr);
  
}

void Renderable::cleanup_swap_chain() {
  vkDestroyPipeline(app_context_->device(), graphics_pipeline_, nullptr);
  vkDestroyPipelineLayout(app_context_->device(), pipeline_layout_, nullptr);

  destroy_uniform_buffers();
  destroy_storage_buffers();

  vkDestroyDescriptorPool(app_context_->device(), descriptor_pool_, nullptr);
}

void Renderable::cleanup() {
  vkDestroyDescriptorSetLayout(app_context_->device(), descriptor_set_layout_,
                               nullptr);

  vkDestroyBuffer(app_context_->device(), index_buffer_, nullptr);
  vkFreeMemory(app_context_->device(), index_buffer_memory_, nullptr);

  vkDestroyBuffer(app_context_->device(), vertex_buffer_, nullptr);
  vkFreeMemory(app_context_->device(), vertex_buffer_memory_, nullptr);

  vkDestroyBuffer(app_context_->device(), staging_index_buffer_, nullptr);
  vkFreeMemory(app_context_->device(), staging_index_buffer_memory_, nullptr);

  vkDestroyBuffer(app_context_->device(), staging_vertex_buffer_, nullptr);
  vkFreeMemory(app_context_->device(), staging_vertex_buffer_memory_, nullptr);
}

void Renderable::record_this_frame_commands(VkCommandBuffer command_buffer) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphics_pipeline_);

  VkBuffer vertex_buffer_s[] = {vertex_buffer_};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffer_s, offsets);

  vkCmdBindIndexBuffer(command_buffer, index_buffer_, 0, VK_INDEX_TYPE_UINT32);

  vkCmdBindDescriptorSets(
      command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1,
      &descriptor_set_ , 0,
      nullptr);

  if (indexed_) {
    vkCmdDrawIndexed(command_buffer, config_.indices_count, 1, 0, 0, 0);
  } else {
    vkCmdDraw(command_buffer, config_.vertices_count, 1, 0, 0);
  }
}

void Renderable::resize_storage_buffers(int new_ssbo_size) {
  if (new_ssbo_size == config_.ssbo_size) {
    return;
  }
  destroy_storage_buffers();
  vkDestroyDescriptorPool(app_context_->device(), descriptor_pool_, nullptr);
  config_.ssbo_size = new_ssbo_size;
  create_storage_buffers();
  create_descriptor_pool();
  create_descriptor_sets();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
