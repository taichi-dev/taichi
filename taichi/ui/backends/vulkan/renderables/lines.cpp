#include "lines.h"

#include "taichi/ui/utils/utils.h"

#include "taichi/rhi/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Lines::update_data(const LinesInfo &info) {
  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.has_per_vertex_color);

  curr_width_ = info.width;
}

void Lines::init_lines(AppContext *app_context,
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
      sizeof(Canvas2dUbo),
      0,
      true,
      app_context->config.package_path + "/shaders/Canvas2D_vk_vert.spv",
      app_context->config.package_path + "/shaders/Canvas2D_vk_frag.spv",
      TopologyType::Lines,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Lines::Lines(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_lines(app_context, 4, 6);
}

void Lines::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  glm::vec4 color2(color, use_per_vertex_color ? 1 : 0);

  Canvas2dUbo ubo{};
  ubo.color = color2;
  ubo.wh_invwh = app_context_->get_wh_invwh();
  ubo.radius = 1.0f;

  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Lines::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
}

void Lines::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());
  command_list->set_line_width(curr_width_ * app_context_->config.height);

  if (indexed_) {
    command_list->draw_indexed(config_.indices_count, 0, 0);
  } else {
    command_list->draw(config_.vertices_count, 0);
  }
}

void Lines::cleanup() {
  Renderable::cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
