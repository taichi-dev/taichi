#include "set_image.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "kernels.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void SetImage::update_data(const SetImageInfo &info) {
  const FieldInfo &img = info.img;
  if (img.shape.size() != 2) {
    throw std::runtime_error(
        "for set image, the image should have exactly two axis. e,g, "
        "ti.Vector.field(3,ti.u8,(1920,1080) ");
  }
  if ((img.matrix_rows != 3 && img.matrix_rows != 4) || img.matrix_cols != 1) {
    throw std::runtime_error(
        "for set image, the image should either a 3-D vector field (RGB) or a "
        "4D vector field (RGBA) ");
  }
  int new_width = img.shape[0];
  int new_height = img.shape[1];

  if (new_width != width || new_height != height) {
    cleanup_swap_chain();
    cleanup();
    init_set_image(renderer_, new_width, new_height);
  }

  int actual_width = next_power_of_2(width);
  int actual_height = next_power_of_2(height);

  int pixels = width * height;
 
  app_context_->vulkan_device().image_transition(texture_,ImageLayout::shader_read,ImageLayout::transfer_dst);

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = width;
  copy_params.image_extent.y = height;

  if (img.field_source == FieldSource::TaichiCuda) {

    unsigned char *mapped = device_ptr_;

    if (img.dtype == PrimitiveType::u8) {
      copy_to_texture_buffer_cuda((unsigned char *)img.data, mapped, width,
                                 height, actual_width, actual_height,
                                 img.matrix_rows);
    } else if (img.dtype == PrimitiveType::f32) {
      copy_to_texture_buffer_cuda((float *)img.data, mapped, width, height,
                                 actual_width, actual_height, img.matrix_rows);
    } else {
      throw std::runtime_error("for set image, dtype must be u8 or f32");
    }

    app_context_->vulkan_device().buffer_to_image(texture_,gpu_staging_buffer_.get_ptr(0),ImageLayout::transfer_dst,copy_params);

  } else if (img.field_source == FieldSource::TaichiX64) {

    unsigned char *mapped =
        (unsigned char *)app_context_->vulkan_device().map(cpu_staging_buffer_);

    if (img.dtype == PrimitiveType::u8) {
      copy_to_texture_buffer_x64((unsigned char *)img.data, mapped, width,
                                 height, actual_width, actual_height,
                                 img.matrix_rows);
    } else if (img.dtype == PrimitiveType::f32) {
      copy_to_texture_buffer_x64((float *)img.data, mapped, width, height,
                                 actual_width, actual_height, img.matrix_rows);
    } else {
      throw std::runtime_error("for set image, dtype must be u8 or f32");
    }

    app_context_->vulkan_device().unmap(gpu_staging_buffer_);
    app_context_->vulkan_device().buffer_to_image(texture_,cpu_staging_buffer_.get_ptr(0),ImageLayout::transfer_dst,copy_params);

    
  } else {
    throw std::runtime_error("unsupported field source");
  }

  app_context_->vulkan_device().image_transition(texture_,ImageLayout::transfer_dst ,ImageLayout::shader_read);
}

SetImage::SetImage(Renderer *renderer) {
  init_set_image(renderer, 1, 1);
}

void SetImage::init_set_image(Renderer *renderer,
                              int img_width,
                              int img_height) {
  RenderableConfig config = {
      6,
      6,
      0,
      0,
      renderer->app_context().config.package_path +
          "/shaders/SetImage_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/SetImage_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, renderer);

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
  params.dimension = ImageDimension::d3D;
  params.format = BufferFormat::rgba8;
  params.initial_layout = ImageLayout::shader_read;
  params.x = width;
  params.y = height;
  params.z = 1;
  params.export_sharing = true;

  texture_ = renderer_->app_context().vulkan_device().create_image(params);



  Device::AllocParams cpu_staging_buffer_params{image_size, true, false, false,
                                            AllocUsage::Uniform};
  cpu_staging_buffer_ = renderer_->app_context().vulkan_device().allocate_memory(
      cpu_staging_buffer_params);

  Device::AllocParams gpu_staging_buffer_params{image_size, false, false, true,
                                            AllocUsage::Uniform};
  gpu_staging_buffer_ = renderer_->app_context().vulkan_device().allocate_memory(
      gpu_staging_buffer_params);

  if (app_context_->config.ti_arch == Arch::cuda) {
    auto [mem,offset,size] = app_context_->vulkan_device().get_vkmemory_offset_size(gpu_staging_buffer_);
  
    auto block_size = VulkanDevice::kMemoryBlockSize;

    device_ptr_ = (unsigned char *)get_memory_pointer(
        mem,block_size,offset,size,
        app_context_->device());
  }

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
        (Vertex *)app_context_->vulkan_device().map(staging_vertex_buffer_);

    memcpy(mapped_vbo, vertices.data(),
           (size_t)config_.vertices_count * sizeof(Vertex));
    app_context_->vulkan_device().unmap(staging_vertex_buffer_);
  }

  app_context_->vulkan_device().memcpy(vertex_buffer_.get_ptr(0),
                                       staging_vertex_buffer_.get_ptr(0),
                                       config_.vertices_count * sizeof(Vertex));
}

void SetImage::update_index_buffer_() {
  const std::vector<uint32_t> indices = {
      0, 1, 2, 3, 4, 5,
  };
  {
    int *mapped_ibo =
        (int *)app_context_->vulkan_device().map(staging_index_buffer_);
    memcpy(mapped_ibo, indices.data(),
           (size_t)config_.indices_count * sizeof(int));
    app_context_->vulkan_device().unmap(staging_index_buffer_);
  }

  app_context_->vulkan_device().memcpy(index_buffer_.get_ptr(0),
                                       staging_index_buffer_.get_ptr(0),
                                       config_.indices_count * sizeof(int));

  indexed_ = true;
}

void SetImage::create_bindings() {
  Renderable::create_bindings();
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->image(0, 0, texture_, {});
}

void SetImage::cleanup() {
  Renderable::cleanup();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
