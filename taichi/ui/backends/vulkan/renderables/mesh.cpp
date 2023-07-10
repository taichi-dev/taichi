#include "mesh.h"

#include "taichi/ui/utils/utils.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;

Mesh::Mesh(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.ubo_size = sizeof(UBORenderable);
  config.blending = true;
  config.depth = true;
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/Mesh_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/Mesh_vk_vert.spv";
  is_3d_renderable = true;
  Renderable::init(config, app_context);
}

void Mesh::update_data(const MeshInfo &info) {
  num_instances_ = info.num_instances;
  start_instance_ = info.start_instance;

  Renderable::update_data(info.renderable_info);

  // Update instance transform buffer
  size_t correct_mesh_ssbo_size = 0;
  if (info.mesh_attribute_info.has_attribute) {
    auto &attr_field = info.mesh_attribute_info.mesh_attribute;
    if (attr_field.dtype != PrimitiveType::f32) {
      TI_ERROR("Data Type transforms of Matrix Field must be ti.f32");
    }

    correct_mesh_ssbo_size = attr_field.num_elements * sizeof(float);
  }
  resize_mesh_storage_buffers(correct_mesh_ssbo_size);

  if (info.mesh_attribute_info.has_attribute) {
    auto &mesh_attribute = info.mesh_attribute_info.mesh_attribute;

    // If data source is not a host mapped pointer, it is a DeviceAllocation
    // from the same backend as GGUI
    DevicePtr attr_dev_ptr = mesh_attribute.dev_alloc.get_ptr();

    copy_helper(app_context_->prog(), mesh_storage_buffer_->get_ptr(),
                attr_dev_ptr, mesh_staging_storage_buffer_->get_ptr(),
                mesh_ssbo_size_);
  }

  // Update UBO
  {
    UBORenderable ubo;
    ubo.color = info.color;
    ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;
    ubo.two_sided = info.two_sided;
    ubo.has_attribute = info.mesh_attribute_info.has_attribute;
    void *mapped{nullptr};
    RHI_VERIFY(app_context_->device().map(
        uniform_buffer_renderable_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_->device().unmap(*uniform_buffer_renderable_);
  }
}

void Mesh::update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr) {
  lights_ssbo_ptr = ssbo_ptr;
  scene_ubo_ptr = ubo_ptr;
}

void Mesh::record_this_frame_commands(taichi::lang::CommandList *command_list) {
  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vertex_buffer_->get_ptr(), 0);
  if (index_buffer_) {
    raster_state->index_buffer(index_buffer_->get_ptr(), 32);
  }

  resource_set_->buffer(0, uniform_buffer_renderable_->get_ptr(0));
  resource_set_->buffer(1, scene_ubo_ptr);
  resource_set_->rw_buffer(2, lights_ssbo_ptr);
  resource_set_->rw_buffer(3, mesh_storage_buffer_->get_ptr(0));

  command_list->bind_pipeline(pipeline_);
  command_list->bind_raster_resources(raster_state.get());
  command_list->bind_shader_resources(resource_set_.get());

  if (indexed_) {
    command_list->draw_indexed_instance(
        config_.draw_index_count, num_instances_, config_.draw_first_vertex,
        config_.draw_first_index, start_instance_);
  } else {
    command_list->draw_instance(config_.draw_vertex_count, num_instances_,
                                config_.draw_first_vertex, start_instance_);
  }
}

void Mesh::resize_mesh_storage_buffers(size_t ssbo_size) {
  if (mesh_storage_buffer_ != nullptr && ssbo_size == mesh_ssbo_size_) {
    return;
  }

  mesh_ssbo_size_ = ssbo_size;
  size_t alloc_size = std::max(4 * 4 * sizeof(float), ssbo_size);

  create_buffer_with_staging(app_context_->device(), alloc_size,
                             AllocUsage::Storage, mesh_storage_buffer_,
                             mesh_staging_storage_buffer_);
}

}  // namespace vulkan

}  // namespace taichi::ui
