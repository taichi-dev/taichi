#include "scene_lines.h"

#include "taichi/ui/utils/utils.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void SceneLines::update_data(const SceneLinesInfo &info, const Scene &scene) {
  Renderable::update_data(info.renderable_info);

  update_ubo(info, scene);

  curr_width_ = info.width;
}

void SceneLines::init_scene_lines(AppContext *app_context,
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
      app_context->config.package_path + "/shaders/SceneLines_vk_vert.spv",
      app_context->config.package_path + "/shaders/SceneLines_vk_frag.spv",
      TopologyType::Lines,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

SceneLines::SceneLines(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_scene_lines(app_context, 4, 6);
}

void SceneLines::update_ubo(const SceneLinesInfo &info, const Scene &scene) {
  UniformBufferObject ubo{};
  ubo.scene = scene.current_ubo_;
  ubo.color = info.color;
  ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;
  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void SceneLines::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
  binder->rw_buffer(0, 1, storage_buffer_);
}

void SceneLines::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());
  command_list->set_line_width(curr_width_);

  if (indexed_) {
    command_list->draw_indexed(config_.draw_index_count,
                               config_.draw_first_vertex,
                               config_.draw_first_index);
  } else {
    command_list->draw(config_.draw_vertex_count, config_.draw_first_vertex);
  }
}

void SceneLines::cleanup() {
  Renderable::cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
