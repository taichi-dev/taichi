#include "particles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

Particles::Particles(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.ubo_size = sizeof(UBORenderable);
  config.depth = true;
  config.blending = true;
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/Particles_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/Particles_vk_vert.spv";
  config.vertex_input_rate_instance = true;  // point instancing
  is_3d_renderable = true;

  Renderable::init(config, app_context);
}

void Particles::update_data(const ParticlesInfo &info) {
  Renderable::update_data(info.renderable_info);

  // Update UBO
  {
    UBORenderable ubo;
    ubo.color = glm::vec4(info.color, 1);
    ubo.radius = info.radius;
    ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;
    ubo.use_per_vertex_radius = info.renderable_info.has_per_vertex_radius;

    void *mapped{nullptr};
    RHI_VERIFY(app_context_->device().map(
        uniform_buffer_renderable_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_->device().unmap(*uniform_buffer_renderable_);
  }
}

void Particles::update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr) {
  lights_ssbo_ptr = ssbo_ptr;
  scene_ubo_ptr = ubo_ptr;
}

void Particles::record_this_frame_commands(CommandList *command_list) {
  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vertex_buffer_->get_ptr(0), 0);

  resource_set_->buffer(0, uniform_buffer_renderable_->get_ptr(0));
  resource_set_->buffer(1, scene_ubo_ptr);
  resource_set_->rw_buffer(2, lights_ssbo_ptr);

  command_list->bind_pipeline(pipeline_);
  command_list->bind_raster_resources(raster_state.get());
  command_list->bind_shader_resources(resource_set_.get());

  // We draw num_particles * 6, 6 forms a quad
  // The `first_instance` should then instead set with `draw_first_vertex`,
  // and the first index always need to be 0
  command_list->draw_instance(/*num_verticies=*/6,
                              /*num_instances=*/config_.draw_vertex_count,
                              /*start_vertex=*/0,
                              /*start_instance=*/config_.draw_first_vertex);
}

}  // namespace vulkan

}  // namespace taichi::ui
