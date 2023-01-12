#include "particles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

Particles::Particles(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_particles(app_context, /*vertices_count=*/1, vbo_attrs);
}

void Particles::update_ubo(glm::vec3 color,
                           bool use_per_vertex_color,
                           float radius,
                           const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = glm::vec4(color, 1);
  ubo.radius = radius;
  ubo.window_width = app_context_->config.width;
  ubo.window_height = app_context_->config.height;
  ubo.tan_half_fov = tanf(glm::radians(scene.camera_.fov) / 2);
  ubo.use_per_vertex_color = use_per_vertex_color;

  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Particles::update_data(const ParticlesInfo &info, const Scene &scene) {
  Renderable::update_data(info.renderable_info);
  size_t correct_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);
  if (config_.ssbo_size != correct_ssbo_size) {
    resize_storage_buffers(correct_ssbo_size);
    create_bindings();
  }
  {
    void *mapped{nullptr};
    TI_ASSERT(app_context_->device().map(storage_buffer_, &mapped) ==
              RhiResult::success);
    memcpy(mapped, scene.point_lights_.data(), correct_ssbo_size);
    app_context_->device().unmap(storage_buffer_);
  }

  update_ubo(info.color, info.renderable_info.has_per_vertex_color, info.radius,
             scene);
}

void Particles::init_particles(AppContext *app_context,
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
      1,
      true,
      app_context->config.package_path + "/shaders/Particles_vk_vert.spv",
      app_context->config.package_path + "/shaders/Particles_vk_frag.spv",
      TopologyType::Triangles,  // We use two triangles to draw out a quad
      PolygonMode::Fill,
      vbo_attrs,
      true  // point instancing
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

void Particles::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_raster_resources(raster_state_.get());
  command_list->bind_shader_resources(resource_set_.get());

  // We draw num_particles * 6, 6 forms a quad
  if (indexed_) {
    command_list->draw_indexed_instance(6, config_.draw_index_count,
                                        config_.draw_first_vertex,
                                        config_.draw_first_index);
  } else {
    command_list->draw_instance(6, config_.draw_vertex_count,
                                config_.draw_first_vertex);
  }
}

void Particles::create_bindings() {
  Renderable::create_bindings();
  resource_set_->buffer(0, uniform_buffer_);
  resource_set_->rw_buffer(1, storage_buffer_);
}

}  // namespace vulkan

}  // namespace taichi::ui
