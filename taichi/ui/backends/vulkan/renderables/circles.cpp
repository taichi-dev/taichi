#include "circles.h"

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

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
      sizeof(Canvas2dUbo),
      0,
      true,
      app_context->config.package_path + "/shaders/Canvas2D_vk_vert.spv",
      app_context->config.package_path + "/shaders/Canvas2D_vk_frag.spv",
      TopologyType::Points,
      PolygonMode::Fill,
      vbo_attrs,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Circles::Circles(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_circles(app_context, /*vertices_count=*/1, vbo_attrs);
}

void Circles::update_ubo(glm::vec3 color,
                         bool use_per_vertex_color,
                         float radius) {
  glm::vec4 color2(color, use_per_vertex_color ? 1 : 0);

  glm::vec4 wh_invwh = app_context_->get_wh_invwh();

  Canvas2dUbo ubo{};
  ubo.color = color2;
  ubo.wh_invwh = wh_invwh;
  ubo.radius = radius * wh_invwh.y;

  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Circles::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
