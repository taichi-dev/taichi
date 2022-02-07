#include "set_image.h"

#include "taichi/program/program.h"
#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

int SetImage::get_correct_dimension(int dimension) {
  if (app_context_->config.is_packed_mode) {
    return dimension;
  } else {
    return next_power_of_2(dimension);
  }
}

void SetImage::update_ubo(float x_factor, float y_factor) {
  UniformBufferObject ubo = {x_factor, y_factor};
  void *mapped = app_context_->device().map(uniform_buffer_);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void SetImage::update_data(const SetImageInfo &info) {
  Program *prog = app_context_->prog();
  prog->synchronize();

  const FieldInfo &img = info.img;

  int new_width = get_correct_dimension(img.shape[0]);
  int new_height = get_correct_dimension(img.shape[1]);

  if (new_width != width || new_height != height) {
    destroy_texture();
    free_buffers();
    init_set_image(app_context_, new_width, new_height);
  }

  update_ubo(img.shape[0] / (float)new_width, img.shape[1] / (float)new_height);

  int pixels = width * height;

  app_context_->device().image_transition(texture_, ImageLayout::shader_read,
                                          ImageLayout::transfer_dst);

  DevicePtr img_dev_ptr = get_device_ptr(prog, img.snode);
  uint64_t img_size = pixels * 4;

  Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
      gpu_staging_buffer_.get_ptr(), img_dev_ptr, img_size);
  if (memcpy_cap == Device::MemcpyCapability::Direct) {
    Device::memcpy_direct(gpu_staging_buffer_.get_ptr(), img_dev_ptr, img_size);
  } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
    Device::memcpy_via_staging(gpu_staging_buffer_.get_ptr(),
                               cpu_staging_buffer_.get_ptr(), img_dev_ptr,
                               img_size);
  } else {
    TI_NOT_IMPLEMENTED;
  }

  BufferImageCopyParams copy_params;
  // these are flipped because taichi is y-major and vulkan is x-major
  copy_params.image_extent.x = height;
  copy_params.image_extent.y = width;

  auto stream = app_context_->device().get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  cmd_list->buffer_to_image(texture_, gpu_staging_buffer_.get_ptr(0),
                            ImageLayout::transfer_dst, copy_params);

  cmd_list->image_transition(texture_, ImageLayout::transfer_dst,
                             ImageLayout::shader_read);
  stream->submit_synced(cmd_list.get());
}

SetImage::SetImage(AppContext *app_context) {
  init_set_image(app_context, 1, 1);
}

void SetImage::init_set_image(AppContext *app_context,
                              int img_width,
                              int img_height) {
  RenderableConfig config = {
      6,
      6,
      6,
      6,
      sizeof(UniformBufferObject),
      0,
      app_context->config.package_path + "/shaders/SetImage_vk_vert.spv",
      app_context->config.package_path + "/shaders/SetImage_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, app_context);

  width = img_width;
  height = img_height;

  create_texture();

  Renderable::init_render_resources();

  update_vertex_buffer_();
  update_index_buffer_();
}

void SetImage::create_texture() {
  size_t image_size = width * height * 4;

  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::rgba8;
  params.initial_layout = ImageLayout::shader_read;
  // these are flipped because taichi is y-major and vulkan is x-major
  params.x = height;
  params.y = width;
  params.z = 1;
  params.export_sharing = true;

  texture_ = app_context_->device().create_image(params);

  Device::AllocParams cpu_staging_buffer_params{image_size, true, false, false,
                                                AllocUsage::Uniform};
  cpu_staging_buffer_ =
      app_context_->device().allocate_memory(cpu_staging_buffer_params);

  Device::AllocParams gpu_staging_buffer_params{
      image_size, false, false, app_context_->requires_export_sharing(),
      AllocUsage::Uniform};
  gpu_staging_buffer_ =
      app_context_->device().allocate_memory(gpu_staging_buffer_params);
}

void SetImage::destroy_texture() {
  app_context_->device().destroy_image(texture_);
  app_context_->device().dealloc_memory(cpu_staging_buffer_);
  app_context_->device().dealloc_memory(gpu_staging_buffer_);
}

void SetImage::update_vertex_buffer_() {
  const std::vector<Vertex> vertices = {
      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{-1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},

      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 1.f, 1.f}},
  };

  {
    Vertex *mapped_vbo =
        (Vertex *)app_context_->device().map(staging_vertex_buffer_);

    memcpy(mapped_vbo, vertices.data(),
           (size_t)config_.vertices_count * sizeof(Vertex));
    app_context_->device().unmap(staging_vertex_buffer_);
  }

  app_context_->device().memcpy_internal(
      vertex_buffer_.get_ptr(0), staging_vertex_buffer_.get_ptr(0),
      config_.vertices_count * sizeof(Vertex));
}

void SetImage::update_index_buffer_() {
  const std::vector<uint32_t> indices = {
      0, 1, 2, 3, 4, 5,
  };
  {
    int *mapped_ibo = (int *)app_context_->device().map(staging_index_buffer_);
    memcpy(mapped_ibo, indices.data(),
           (size_t)config_.indices_count * sizeof(int));
    app_context_->device().unmap(staging_index_buffer_);
  }

  app_context_->device().memcpy_internal(index_buffer_.get_ptr(0),
                                         staging_index_buffer_.get_ptr(0),
                                         config_.indices_count * sizeof(int));

  indexed_ = true;
}

void SetImage::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->image(0, 0, texture_, {});
  binder->buffer(0, 1, uniform_buffer_);
}

void SetImage::cleanup() {
  destroy_texture();
  Renderable::cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
