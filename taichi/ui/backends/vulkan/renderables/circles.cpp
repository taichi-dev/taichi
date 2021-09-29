#include "circles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"

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

void Circles::init_circles(AppContext *app_context, int vertices_count) {
  RenderableConfig config = {
      vertices_count,
      1,
      sizeof(UniformBufferObject),
      0,
      app_context->config.package_path + "/shaders/Circles_vk_vert.spv",
      app_context->config.package_path + "/shaders/Circles_vk_frag.spv",
      TopologyType::Points,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Circles::Circles(AppContext *app_context) {
  init_circles(app_context, 1);
}

void Circles::update_ubo(glm::vec3 color,
                         bool use_per_vertex_color,
                         float radius) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color,
                          radius * app_context_->config.height};

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
