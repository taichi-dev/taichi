#include "mesh.h"

#include "taichi/ui/utils/utils.h"
#include "taichi/runtime/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

Mesh::Mesh(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_mesh(app_context, /*vertices_count=*/3, /*indices_count*/ 3, vbo_attrs);
}

void Mesh::update_ubo(const MeshInfo &info, const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = info.color;
  ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;
  ubo.two_sided = info.two_sided;

  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Mesh::update_data(const MeshInfo &info, const Scene &scene) {
  Renderable::update_data(info.renderable_info);

  size_t correct_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);
  if (config_.ssbo_size != correct_ssbo_size) {
    resize_storage_buffers(correct_ssbo_size);
    create_bindings();
  }
  {
    void *mapped = app_context_->device().map(storage_buffer_);
    memcpy(mapped, scene.point_lights_.data(), correct_ssbo_size);
    app_context_->device().unmap(storage_buffer_);
  }

  update_ubo(info, scene);
}

void Mesh::init_mesh(AppContext *app_context,
                     int vertices_count,
                     int indices_count,
                     VertexAttributes vbo_attrs) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      vertices_count,
      indices_count,
      sizeof(UniformBufferObject),
      1,
      true,
      app_context->config.package_path + "/shaders/Mesh_vk_vert.spv",
      app_context->config.package_path + "/shaders/Mesh_vk_frag.spv",
      TopologyType::Triangles,
      vbo_attrs,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

void Mesh::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
  binder->rw_buffer(0, 1, storage_buffer_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
