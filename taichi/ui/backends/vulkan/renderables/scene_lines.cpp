#include "scene_lines.h"

#include "taichi/ui/utils/utils.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void SceneLines::update_data(const SceneLinesInfo &info, const Scene &scene) {
  Renderable::update_data(info.renderable_info);

  lines_count_ =
      (indexed_ ? config_.indices_count : config_.vertices_count) / 2;

  // Update UBO
  {
    UniformBufferObject ubo{};
    ubo.scene = scene.current_ubo_;
    ubo.color = info.color;
    // FIXME: Why is the width in pixel units?
    ubo.line_width = info.width / float(app_context_->config.height);
    ubo.per_vertex_color_offset = info.renderable_info.has_per_vertex_color
                                      ? offsetof(Vertex, color) / sizeof(float)
                                      : 0;
    ubo.vertex_stride = sizeof(Vertex) / sizeof(float);
    ubo.start_vertex = config_.draw_first_vertex;
    ubo.start_index = config_.draw_first_index;
    ubo.num_vertices = lines_count_ * 2;
    ubo.is_indexed = indexed_ ? 1 : 0;
    ubo.aspect_ratio =
        float(app_context_->config.width) / float(app_context_->config.height);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_->device().map(uniform_buffer_->get_ptr(), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_->device().unmap(*uniform_buffer_);
  }
}

void SceneLines::create_graphics_pipeline() {
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
                                                      sizeof(glm::vec4) * 2,
                                                      /*instance=*/false}};
    // TODO: consider using uint8 for colors and normals
    std::vector<VertexInputAttribute> vertex_attribs;
    vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                              /*format=*/BufferFormat::rgba32f,
                              /*offset=*/0});
    vertex_attribs.push_back({/*location=*/1, /*binding=*/0,
                              /*format=*/BufferFormat::rgba32f,
                              /*offset=*/sizeof(glm::vec4)});

    pipeline_ = app_context_->device().create_raster_pipeline(
        source, raster_params, vertex_inputs, vertex_attribs);
  }

  if (!quad_expand_pipeline_.get()) {
    auto comp_code = read_file(app_context_->config.package_path +
                               "/shaders/SceneLines2quad_vk_comp.spv");
    auto [pipeline, res] = app_context_->device().create_pipeline_unique(
        {PipelineSourceType::spirv_binary, comp_code.data(), comp_code.size(),
         PipelineStageType::compute});
    TI_ASSERT(res == RhiResult::success);
    quad_expand_pipeline_ = std::move(pipeline);
  }
}

SceneLines::SceneLines(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.ubo_size = sizeof(UniformBufferObject);
  config.blending = true;
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/SceneLines_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/SceneLines_vk_vert.spv";
  config.vbo_attrs = vbo_attrs;

  Renderable::init(config, app_context);
}

void SceneLines::record_prepass_this_frame_commands(CommandList *command_list) {
  vbo_translated_.reset();
  ibo_translated_.reset();

  struct TransformedVertex {
    glm::vec4 pos;
    glm::vec4 color;
  };

  vbo_translated_ = app_context_->device().allocate_memory_unique(
      {/*size=*/uint64_t(sizeof(TransformedVertex) * 4 * lines_count_),
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

  resource_set_->rw_buffer(0, vertex_buffer_->get_ptr(0));
  if (index_buffer_) {
    resource_set_->rw_buffer(1, index_buffer_->get_ptr(0));
  } else {
    // Just bind a dummy buffer
    resource_set_->rw_buffer(1, vertex_buffer_->get_ptr(0));
  }
  resource_set_->rw_buffer(2, vbo_translated_->get_ptr(0));
  resource_set_->rw_buffer(3, ibo_translated_->get_ptr(0));
  resource_set_->buffer(4, uniform_buffer_->get_ptr(0));

  command_list->bind_pipeline(quad_expand_pipeline_.get());
  command_list->bind_shader_resources(resource_set_.get());
  command_list->dispatch(int(ceil(lines_count_ / 256.0f)));
  command_list->buffer_barrier(*vbo_translated_);
  command_list->buffer_barrier(*ibo_translated_);
}

void SceneLines::record_this_frame_commands(CommandList *command_list) {
  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vbo_translated_->get_ptr(0), 0);
  raster_state->index_buffer(ibo_translated_->get_ptr(0), 32);

  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_raster_resources(raster_state.get());
  command_list->draw_indexed(lines_count_ * 6, 0, 0);
}

}  // namespace vulkan

}  // namespace taichi::ui
