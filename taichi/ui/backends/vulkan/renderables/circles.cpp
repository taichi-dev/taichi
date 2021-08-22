#include "circles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

void Circles::update_data(const CirclesInfo &info) {
  if (info.renderable_info.vertices.matrix_rows != 2 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Circles vertices requres 2-d vector fields");
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid,
             info.radius);
}

void Circles::init_circles(Renderer *renderer, int vertices_count) {
  RenderableConfig config = {
      vertices_count,
      1,
      sizeof(UniformBufferObject),
      0,
      renderer->app_context().config.package_path +
          "/shaders/Circles_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/Circles_vk_frag.spv",
      TopologyType::Points,
  };

  Renderable::init(config, renderer);
  Renderable::init_render_resources();
}

Circles::Circles(Renderer *renderer) {
  init_circles(renderer, 1);
}

void Circles::update_ubo(glm::vec3 color,
                         bool use_per_vertex_color,
                         float radius) {
  UniformBufferObject ubo{
      color, (int)use_per_vertex_color,
      radius * renderer_->swap_chain().height()};

  void* mapped = renderer_->app_context().vulkan_device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  renderer_->app_context().vulkan_device().unmap(uniform_buffer_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
