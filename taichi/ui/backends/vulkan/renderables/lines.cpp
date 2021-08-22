#include "lines.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/utils/utils.h"

#include "taichi/ui/backends/vulkan/renderables/kernels.h"

#include "taichi/backends/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Lines::update_data(const LinesInfo &info) {
  if (info.renderable_info.vertices.matrix_rows != 2 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Lines vertices requres 2-d vector fields");
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid);

  curr_width_ = info.width;
}

void Lines::record_this_frame_commands(CommandList* command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());
  //TODO: change line width
  if (indexed_) {
    command_list->draw_indexed(config_.indices_count,0,0);
  } else {
    command_list->draw(config_.vertices_count,0);
  }
}

void Lines::init_lines(Renderer *renderer,
                       int vertices_count,
                       int indices_count) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      sizeof(UniformBufferObject),
      0,
      renderer->app_context().config.package_path +
          "/shaders/Lines_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/Lines_vk_frag.spv",
      TopologyType::Lines,
  };

  Renderable::init(config, renderer);
  Renderable::init_render_resources();
}

Lines::Lines(Renderer *renderer) {
  init_lines(renderer, 4, 6);
}

void Lines::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color};

  void* mapped = renderer_->app_context().vulkan_device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  renderer_->app_context().vulkan_device().unmap(uniform_buffer_);
}

void Lines::create_bindings(){
  Renderable::create_bindings();
  ResourceBinder* binder = pipeline_->resource_binder();
  binder->buffer(0,0,uniform_buffer_);
}


void Lines::cleanup() {
  Renderable::cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
