#include "particles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

Particles::Particles(Renderer *renderer) {
  init_particles(renderer, 1);
}

void Particles::update_ubo(glm::vec3 color,
                           bool use_per_vertex_color,
                           float radius,
                           const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = glm::vec4(color, 1);
  ubo.radius = radius;
  ubo.window_width = renderer_->swap_chain().width();
  ubo.window_height = renderer_->swap_chain().height();
  ubo.tan_half_fov = tan(glm::radians(scene.camera_.fov) / 2);
  ubo.use_per_vertex_color = use_per_vertex_color;

  void *mapped = renderer_->app_context().device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  renderer_->app_context().device().unmap(uniform_buffer_);
}

void Particles::update_data(const ParticlesInfo &info, const Scene &scene) {
  if (info.renderable_info.vertices.matrix_rows != 3 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Particles vertices requres 3-d vector fields");
  }

  size_t correct_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);
  if (config_.ssbo_size != correct_ssbo_size) {
    resize_storage_buffers(correct_ssbo_size);
  }
  {
    void *mapped = renderer_->app_context().device().map(storage_buffer_);
    memcpy(mapped, scene.point_lights_.data(), correct_ssbo_size);
    renderer_->app_context().device().unmap(storage_buffer_);
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid,
             info.radius, scene);
}

void Particles::init_particles(Renderer *renderer, int vertices_count) {
  RenderableConfig config = {
      vertices_count,
      vertices_count,
      sizeof(UniformBufferObject),
      1,
      renderer->app_context().config.package_path +
          "/shaders/Particles_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/Particles_vk_frag.spv",
      TopologyType::Points,
  };

  Renderable::init(config, renderer);
  Renderable::init_render_resources();
}

void Particles::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
  binder->rw_buffer(0, 1, storage_buffer_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
