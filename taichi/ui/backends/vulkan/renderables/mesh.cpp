#include "mesh.h"

#include "taichi/ui/utils/utils.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

Mesh::Mesh(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_mesh(app_context, /*vertices_count=*/3, /*indices_count*/ 3, vbo_attrs);
}
void Mesh::cleanup() {
  Renderable::cleanup();
  destroy_mesh_storage_buffers();
}

void Mesh::update_ubo(const MeshInfo &info, const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = info.color;
  ubo.use_per_vertex_color = info.renderable_info.has_per_vertex_color;
  ubo.two_sided = info.two_sided;
  ubo.has_attribute = info.mesh_attribute_info.has_attribute;
  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void Mesh::update_data(const MeshInfo &info, const Scene &scene) {
  num_instances_ = info.num_instances;
  start_instance_ = info.start_instance;

  Renderable::update_data(info.renderable_info);

  size_t correct_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);

  bool is_resize = false;

  if (config_.ssbo_size != correct_ssbo_size) {
    resize_storage_buffers(correct_ssbo_size);
    is_resize = true;
  }

  if (info.mesh_attribute_info.has_attribute) {
    auto &attr_field = info.mesh_attribute_info.mesh_attribute;
    if (attr_field.dtype != PrimitiveType::f32 &&
        attr_field.dtype != PrimitiveType::u32 &&
        attr_field.dtype != PrimitiveType::i32) {
      TI_ERROR(
          "Data Type transforms of Matrix Field must be ti.f32 or ti.u32 or "
          "ti.i32");
    }

    size_t correct_mesh_ssbo_size =
        attr_field.shape[0] * attr_field.matrix_rows * attr_field.matrix_cols *
        data_type_size(attr_field.dtype);

    if (correct_mesh_ssbo_size != mesh_ssbo_size_) {
      resize_mesh_storage_buffers(correct_mesh_ssbo_size);
      is_resize = true;
    }
  }

  if (is_resize) {
    create_bindings();
  }

  {
    void *mapped = app_context_->device().map(storage_buffer_);
    memcpy(mapped, scene.point_lights_.data(), correct_ssbo_size);
    app_context_->device().unmap(storage_buffer_);
  }

  if (info.mesh_attribute_info.has_attribute) {
    Program *prog = app_context_->prog();
    if (prog) {
      prog->flush();
    }

    // If there is no current program, VBO information should be provided
    // directly instead of accessing through the current SNode
    DevicePtr attr_dev_ptr =
        info.mesh_attribute_info.mesh_attribute.dev_alloc.get_ptr();
    if (prog) {
      attr_dev_ptr =
          get_device_ptr(prog, info.mesh_attribute_info.mesh_attribute.snode);
    }
    Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
        mesh_storage_buffer_.get_ptr(), attr_dev_ptr, mesh_ssbo_size_);
    if (memcpy_cap == Device::MemcpyCapability::Direct) {
      Device::memcpy_direct(mesh_storage_buffer_.get_ptr(), attr_dev_ptr,
                            mesh_ssbo_size_);
    } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
      void *ssbo_mapped = app_context_->device().map(mesh_storage_buffer_);
      DeviceAllocation attr_buffer(attr_dev_ptr);
      void *attr_mapped = attr_dev_ptr.device->map(attr_buffer);
      memcpy(ssbo_mapped, attr_mapped, mesh_ssbo_size_);
      app_context_->device().unmap(mesh_storage_buffer_);
      attr_dev_ptr.device->unmap(attr_buffer);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  update_ubo(info, scene);
}

void Mesh::record_this_frame_commands(taichi::lang::CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());

  if (indexed_) {
    command_list->draw_indexed_instance(
        config_.draw_index_count, num_instances_, config_.draw_first_vertex,
        config_.draw_first_index, start_instance_);
  } else {
    command_list->draw_instance(config_.draw_vertex_count, num_instances_,
                                config_.draw_first_vertex, start_instance_);
  }
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
      vertices_count,
      0,
      indices_count,
      0,
      sizeof(UniformBufferObject),
      1,
      true,
      app_context->config.package_path + "/shaders/Mesh_vk_vert.spv",
      app_context->config.package_path + "/shaders/Mesh_vk_frag.spv",
      TopologyType::Triangles,
      PolygonMode::Fill,
      vbo_attrs,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();

  create_mesh_storage_buffers();
}

void Mesh::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->buffer(0, 0, uniform_buffer_);
  binder->rw_buffer(0, 1, storage_buffer_);
  binder->rw_buffer(0, 2, mesh_storage_buffer_);
}

void Mesh::create_mesh_storage_buffers() {
  if (mesh_ssbo_size_ == 0) {
    mesh_ssbo_size_ = 4 * 4 * sizeof(float);
  }
  Device::AllocParams sb_params{mesh_ssbo_size_, true, false, true,
                                AllocUsage::Storage};
  mesh_storage_buffer_ = app_context_->device().allocate_memory(sb_params);
}

void Mesh::destroy_mesh_storage_buffers() {
  app_context_->device().dealloc_memory(mesh_storage_buffer_);
}

void Mesh::resize_mesh_storage_buffers(size_t ssbo_size) {
  if (mesh_ssbo_size_ != 0) {
    destroy_mesh_storage_buffers();
  }
  mesh_ssbo_size_ = ssbo_size;
  create_mesh_storage_buffers();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
