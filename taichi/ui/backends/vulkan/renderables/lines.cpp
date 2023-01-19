#include "lines.h"

#include "taichi/ui/utils/utils.h"

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Lines::init_lines(AppContext *app_context,
                       int vertices_count,
                       int indices_count) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      vertices_count,
      indices_count,
      vertices_count,
      0,
      indices_count,
      0,
      sizeof(UniformBufferObject),
      0,
      true,
      app_context->config.package_path + "/shaders/Lines_vk_vert.spv",
      app_context->config.package_path + "/shaders/Lines_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

void Lines::create_graphics_pipeline() {
  if (!pipeline_.get()) {
    auto vert_code = read_file(config_.vertex_shader_path);
    auto frag_code = read_file(config_.fragment_shader_path);

    std::vector<PipelineSourceDesc> source(2);
    source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
                 frag_code.size(), PipelineStageType::fragment};
    source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
                 vert_code.size(), PipelineStageType::vertex};

    RasterParams raster_params;
    raster_params.prim_topology = TopologyType::Triangles;
    raster_params.polygon_mode = config_.polygon_mode;
    raster_params.depth_test = true;
    raster_params.depth_write = true;

    if (config_.blending) {
      raster_params.blending.push_back(BlendingParams());
    }

    std::vector<VertexInputBinding> vertex_inputs = {{/*binding=*/0,
                                                      sizeof(float) * 4,
                                                      /*instance=*/false}};
    // TODO: consider using uint8 for colors and normals
    std::vector<VertexInputAttribute> vertex_attribs;
    vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                              /*format=*/BufferFormat::rg32f,
                              /*offset=*/0});
    vertex_attribs.push_back({/*location=*/1, /*binding=*/0,
                              /*format=*/BufferFormat::r32u,
                              /*offset=*/sizeof(float) * 2});

    pipeline_ = app_context_->device().create_raster_pipeline(
        source, raster_params, vertex_inputs, vertex_attribs);
  }

  if (!quad_expand_pipeline_.get()) {
    auto comp_code = read_file(app_context_->config.package_path +
                               "/shaders/lines2quad_vk_comp.spv");
    auto [pipeline, res] = app_context_->device().create_pipeline_unique(
        {PipelineSourceType::spirv_binary, comp_code.data(), comp_code.size(),
         PipelineStageType::compute});
    TI_ASSERT(res == RhiResult::success);
    quad_expand_pipeline_ = std::move(pipeline);
  }
}

Lines::Lines(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_lines(app_context, 4, 6);
}

void Lines::update_data(const LinesInfo &info) {
  Renderable::update_data(info.renderable_info);

  lines_count_ =
      (indexed_ ? config_.indices_count : config_.vertices_count) / 2;

  update_ubo(info.color, info.renderable_info.has_per_vertex_color, info.width);
}

void Lines::update_ubo(glm::vec3 color,
                       bool use_per_vertex_color,
                       float line_width) {
  UniformBufferObject ubo{};
  ubo.color = color;
  ubo.line_width = line_width;
  ubo.per_vertex_color_offset =
      use_per_vertex_color ? offsetof(Vertex, color) / sizeof(float) : 0;
  ubo.vertex_stride = config_.vbo_size() / sizeof(float);
  ubo.start_vertex = 0;
  ubo.start_index = 0;
  ubo.num_vertices = lines_count_ * 2;
  ubo.is_indexed = indexed_ ? 1 : 0;
  ubo.aspect_ratio =
      float(app_context_->config.width) / float(app_context_->config.height);

  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Lines::create_bindings() {
  if (!resource_set_) {
    resource_set_ = app_context_->device().create_resource_set_unique();
  }
  if (!raster_state_) {
    raster_state_ = app_context_->device().create_raster_resources_unique();
  }
}

void Lines::record_prepass_this_frame_commands(CommandList *command_list) {
  vbo_translated_.reset();
  ibo_translated_.reset();

  vbo_translated_ = app_context_->device().allocate_memory_unique(
      {/*size=*/uint64_t(sizeof(glm::vec4) * 4 * lines_count_),
       /*host_write=*/false,
       /*host_read=*/false,
       /*export_sharing=*/false,
       /*usage=*/AllocUsage::Storage | AllocUsage::Vertex});

  ibo_translated_ = app_context_->device().allocate_memory_unique(
      {/*size=*/uint64_t(sizeof(int) * 6 * lines_count_),
       /*host_write=*/false,
       /*host_read=*/false,
       /*export_sharing=*/false,
       /*usage=*/AllocUsage::Storage | AllocUsage::Index});

  raster_state_->vertex_buffer(vbo_translated_->get_ptr(0), 0);
  raster_state_->index_buffer(ibo_translated_->get_ptr(0), 32);

  resource_set_->rw_buffer(0, vertex_buffer_.get_ptr(0));
  resource_set_->rw_buffer(1, index_buffer_.get_ptr(0));
  resource_set_->rw_buffer(2, vbo_translated_->get_ptr(0));
  resource_set_->rw_buffer(3, ibo_translated_->get_ptr(0));
  resource_set_->buffer(4, uniform_buffer_.get_ptr(0));

  command_list->bind_pipeline(quad_expand_pipeline_.get());
  command_list->bind_shader_resources(resource_set_.get());
  command_list->dispatch(int(ceil(lines_count_ / 256.0f)));
  command_list->buffer_barrier(*vbo_translated_);
  command_list->buffer_barrier(*ibo_translated_);
}

void Lines::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_raster_resources(raster_state_.get());
  command_list->draw_indexed(lines_count_ * 6, 0, 0);
}

void Lines::cleanup() {
  Renderable::cleanup();

  vbo_translated_.reset();
  ibo_translated_.reset();
}

}  // namespace vulkan

}  // namespace taichi::ui
