#include "triangles.h"

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Triangles::update_data(const TrianglesInfo &info) {
  Renderable::update_data(info.renderable_info);

  UniformBufferObject ubo{info.color,
                          (int)info.renderable_info.has_per_vertex_color};

  void *mapped{nullptr};
  RHI_VERIFY(app_context_->device().map(uniform_buffer_renderable_->get_ptr(0),
                                        &mapped));
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(*uniform_buffer_renderable_);
}

Triangles::Triangles(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.ubo_size = sizeof(UniformBufferObject);
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/Triangles_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/Triangles_vk_vert.spv";
  config.blending = true;

  Renderable::init(config, app_context);
}

}  // namespace vulkan

}  // namespace taichi::ui
