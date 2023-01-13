#include "lines.h"

#include "taichi/ui/utils/utils.h"

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::ui {

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
      sizeof(UniformBufferObject),
      0,
      true,
      app_context->config.package_path + "/shaders/Lines_vk_vert.spv",
      app_context->config.package_path + "/shaders/Lines_vk_frag.spv",
      TopologyType::Lines,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Lines::Lines(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_lines(app_context, 4, 6);
}

void Lines::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color};

  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Lines::create_bindings() {
  Renderable::create_bindings();
  resource_set_->buffer(0, uniform_buffer_);
}

void Lines::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_raster_resources(raster_state_.get());
  command_list->bind_shader_resources(resource_set_.get());
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

}  // namespace taichi::ui
