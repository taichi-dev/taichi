#include "particles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

Particles::Particles(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.ubo_size = sizeof(UniformBufferObject);
  config.blending = true;
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/Particles_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/Particles_vk_vert.spv";
  config.vbo_attrs = vbo_attrs;
  config.vertex_input_rate_instance = true;  // point instancing

  Renderable::init(config, app_context);
}

void Particles::update_data(const ParticlesInfo &info, const Scene &scene) {
  Renderable::update_data(info.renderable_info);

  // Update SSBO
  {
    size_t new_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);
    resize_storage_buffers(new_ssbo_size);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_->device().map(storage_buffer_->get_ptr(), &mapped));
    memcpy(mapped, scene.point_lights_.data(), new_ssbo_size);
    app_context_->device().unmap(*storage_buffer_);
  }

  // Update UBO
  {
    UniformBufferObject ubo;
    ubo.scene = scene.current_ubo_;
    ubo.color = glm::vec4(info.color, 1);
    ubo.radius = info.radius;
    ubo.window_width = app_context_->config.width;
    ubo.window_height = app_context_->config.height;
    ubo.tan_half_fov = tanf(glm::radians(scene.camera_.fov) / 2);
    ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;

    void *mapped{nullptr};
    RHI_VERIFY(
        app_context_->device().map(uniform_buffer_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_->device().unmap(*uniform_buffer_);
  }
}

void Particles::record_this_frame_commands(CommandList *command_list) {
  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vertex_buffer_->get_ptr(0), 0);

  resource_set_->buffer(0, uniform_buffer_->get_ptr(0));
  resource_set_->rw_buffer(1, storage_buffer_->get_ptr(0));

  command_list->bind_pipeline(pipeline_.get());
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
