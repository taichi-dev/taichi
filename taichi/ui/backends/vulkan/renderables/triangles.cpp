#include "triangles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Triangles::update_data(const TrianglesInfo &info) {
  if (info.renderable_info.vertices.matrix_rows != 2 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Triangles vertices requres 2-d vector fields");
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid);
}

void Triangles::init_triangles(Renderer *renderer,
                               int vertices_count,
                               int indices_count) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      sizeof(UniformBufferObject),
      0,
      renderer->app_context().config.package_path +
          "/shaders/Triangles_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/Triangles_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, renderer);
  Renderable::init_render_resources();
}

Triangles::Triangles(Renderer *renderer) {
  init_triangles(renderer, 3, 3);
}

void Triangles::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color};

  void *mapped = renderer_->app_context().device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  renderer_->app_context().device().unmap(uniform_buffer_);
}

void Triangles::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
