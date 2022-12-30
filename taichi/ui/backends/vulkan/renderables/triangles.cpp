#include "triangles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Triangles::update_data(const TrianglesInfo &info) {
  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.has_per_vertex_color);
}

void Triangles::init_triangles(AppContext *app_context,
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
      app_context->config.package_path + "/shaders/Triangles_vk_vert.spv",
      app_context->config.package_path + "/shaders/Triangles_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Triangles::Triangles(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_triangles(app_context, 3, 3);
}

void Triangles::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color};

  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Triangles::create_bindings() {
  Renderable::create_bindings();
  resource_set_->buffer(0, uniform_buffer_);
}

}  // namespace vulkan

}  // namespace taichi::ui
