#include "circles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Circles::update_data(const CirclesInfo &info) {
  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.has_per_vertex_color,
             info.radius);
}

void Circles::init_circles(AppContext *app_context,
                           int vertices_count,
                           VertexAttributes vbo_attrs) {
  RenderableConfig config = {
      vertices_count,
      1,
      vertices_count,
      1,
      vertices_count,
      0,
      1,
      0,
      sizeof(UniformBufferObject),
      0,
      true,
      app_context->config.package_path + "/shaders/Circles_vk_vert.spv",
      app_context->config.package_path + "/shaders/Circles_vk_frag.spv",
      TopologyType::Triangles,
      PolygonMode::Fill,
      vbo_attrs,
      true  // Point instanced quads
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Circles::Circles(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_circles(app_context, /*vertices_count=*/1, vbo_attrs);
}

void Circles::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_raster_resources(raster_state_.get());
  command_list->bind_shader_resources(resource_set_.get());

  // We draw num_particles * 6, 6 forms a quad
  // The `first_instance` should then instead set with `draw_first_vertex`,
  // and the first index always need to be 0
  command_list->draw_instance(/*num_verticies=*/6,
                              /*num_instances=*/config_.draw_vertex_count,
                              /*start_vertex=*/0,
                              /*start_instance=*/config_.draw_first_vertex);
}

void Circles::update_ubo(glm::vec3 color,
                         bool use_per_vertex_color,
                         float radius) {
  UniformBufferObject ubo{
      color, (int)use_per_vertex_color, radius,
      radius * app_context_->config.width / app_context_->config.height};

  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Circles::create_bindings() {
  Renderable::create_bindings();
  resource_set_->buffer(0, uniform_buffer_);
}

}  // namespace vulkan

}  // namespace taichi::ui
