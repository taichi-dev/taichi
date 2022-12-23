#include "set_image.h"

#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

namespace taichi::ui {

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

void SetImage::update_ubo(float x_factor, float y_factor, bool transpose) {
  UniformBufferObject ubo = {x_factor, y_factor, int(transpose)};
  void *mapped{nullptr};
  TI_ASSERT(app_context_->device().map(uniform_buffer_, &mapped) ==
            RhiResult::success);
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(uniform_buffer_);
}

void SetImage::update_data(const SetImageInfo &info) {
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  StreamSemaphore sema = nullptr;

  const FieldInfo &img = info.img;

  // Support configuring the internal image based on the data type of the field
  // info.  We assume that the internal image is 4 channels and allow the user
  // to configure either a classic RGBA8 (u8) or RGBA32F (f32). The latter is
  // useful for target that support this texture type as it allows to re-use the
  // result of a kernel directly without normalizing the value from [0; 1] to
  // [0; 255]
  //
  // @TODO: Make the number of channel configurable?
  TI_ASSERT(img.dtype == taichi::lang::PrimitiveType::f32 ||
            img.dtype == taichi::lang::PrimitiveType::u32);
  if (img.dtype == taichi::lang::PrimitiveType::u32) {
    texture_dtype_ = taichi::lang::PrimitiveType::u8;
  } else {
    texture_dtype_ = img.dtype;
  }

  int new_width = get_correct_dimension(img.shape[0]);
  int new_height = get_correct_dimension(img.shape[1]);

  BufferFormat fmt = BufferFormat::rgba8;
  if (texture_dtype_ == taichi::lang::PrimitiveType::f32) {
    fmt = BufferFormat::rgba32f;
  }

  if (new_width != width || new_height != height || fmt != format_) {
    destroy_texture();
    free_buffers();
    init_set_image(app_context_, new_width, new_height, fmt);
  }

  update_ubo(img.shape[0] / (float)new_width, img.shape[1] / (float)new_height,
             true);

  int pixels = width * height;

  uint64_t img_size = pixels * data_type_size(texture_dtype_) * 4;

  // If there is no current program, VBO information should be provided directly
  // instead of accessing through the current SNode
  DevicePtr img_dev_ptr = info.img.dev_alloc.get_ptr();
  if (prog) {
    img_dev_ptr = get_device_ptr(prog, img.snode);
    if (img_dev_ptr.device != &app_context_->device()) {
      sema = prog->flush();
    }
  }
  bool use_enqueued_op =
      prog && (img_dev_ptr.device == &app_context_->device());

  Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
      gpu_staging_buffer_.get_ptr(), img_dev_ptr, img_size);
  if (memcpy_cap == Device::MemcpyCapability::Direct) {
    // If it's the same device, we do not use the staging buffer and directly
    // copy from the src ptr to the image in the `copy_op`
    if (!use_enqueued_op) {
      Device::memcpy_direct(gpu_staging_buffer_.get_ptr(), img_dev_ptr,
                            img_size);
    }
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

  DevicePtr src_ptr =
      use_enqueued_op ? img_dev_ptr : gpu_staging_buffer_.get_ptr(0);

  auto copy_op = [texture = this->texture_, src_ptr, copy_params](
                     Device *device, CommandList *cmdlist) {
    cmdlist->image_transition(texture, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->buffer_barrier(src_ptr);
    cmdlist->buffer_to_image(texture, src_ptr, ImageLayout::transfer_dst,
                             copy_params);
    cmdlist->image_transition(texture, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
  };

  if (use_enqueued_op) {
    prog->enqueue_compute_op_lambda(copy_op, {});
  } else {
    auto stream = app_context_->device().get_graphics_stream();
    auto cmd_list = stream->new_command_list();
    copy_op(&app_context_->device(), cmd_list.get());
    if (sema) {
      stream->submit(cmd_list.get(), {sema});
    } else {
      stream->submit(cmd_list.get());
    }
  }
}

void SetImage::update_data(Texture *tex) {
  Program *prog = app_context_->prog();

  auto shape = tex->get_size();
  auto fmt = tex->get_buffer_format();

  TI_ASSERT_INFO(shape[2] == 1,
                 "Must be a 2D image! Received image shape: {}x{}x{}", shape[0],
                 shape[1], shape[2]);

  // Reminder: y/x is flipped in Taichi. I would like to use the correct
  // orientation, but we have existing code already using the previous
  // convention
  if (shape[1] != width || shape[0] != height || fmt != format_) {
    destroy_texture();
    free_buffers();
    init_set_image(app_context_, shape[1], shape[0], fmt);
  }

  update_ubo(1.0f, 1.0f, false);

  ImageCopyParams copy_params;
  copy_params.width = shape[0];
  copy_params.height = shape[1];
  copy_params.depth = shape[2];

  DeviceAllocation src_alloc = tex->get_device_allocation();
  auto copy_op = [texture = this->texture_, src_alloc, copy_params](
                     Device *device, CommandList *cmdlist) {
    cmdlist->image_transition(texture, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->copy_image(texture, src_alloc, ImageLayout::transfer_dst,
                        ImageLayout::transfer_src, copy_params);
    cmdlist->image_transition(texture, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
  };

  // In the current state if we called this direct image update data method, we
  // gurantee to have a program.
  // FIXME: However, if we don't have a Program, where does the layout come
  // from?
  if (prog) {
    prog->enqueue_compute_op_lambda(
        copy_op, {ComputeOpImageRef{src_alloc, ImageLayout::transfer_src,
                                    ImageLayout::transfer_src}});
  } else {
    auto stream = app_context_->device().get_graphics_stream();
    auto cmd_list = stream->new_command_list();
    copy_op(&app_context_->device(), cmd_list.get());
    stream->submit(cmd_list.get());
  }
}

SetImage::SetImage(AppContext *app_context, VertexAttributes vbo_attrs) {
  init_set_image(app_context, 1, 1, BufferFormat::rgba8);
}

void SetImage::init_set_image(AppContext *app_context,
                              int img_width,
                              int img_height,
                              taichi::lang::BufferFormat format) {
  RenderableConfig config = {
      6,
      6,
      6,
      6,
      6,
      0,
      6,
      0,
      sizeof(UniformBufferObject),
      0,
      false,
      app_context->config.package_path + "/shaders/SetImage_vk_vert.spv",
      app_context->config.package_path + "/shaders/SetImage_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, app_context);

  this->width = img_width;
  this->height = img_height;
  format_ = format;

  create_texture();

  Renderable::init_render_resources();

  update_vertex_buffer();
  update_index_buffer();
}

void SetImage::create_texture() {
  size_t image_size = width * height * data_type_size(texture_dtype_) * 4;

  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = format_;
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

void SetImage::update_vertex_buffer() {
  const std::vector<Vertex> vertices = {
      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{-1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},

      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 1.f, 1.f}},
  };
  // Our actual VBO might only use the first several attributes in `Vertex`,
  // therefore this slicing & copying for each Vertex.
  {
    void *mapped_vbo{nullptr};
    TI_ASSERT(app_context_->device().map(staging_vertex_buffer_, &mapped_vbo) ==
              RhiResult::success);
    for (int i = 0; i < vertices.size(); ++i) {
      const char *src = reinterpret_cast<const char *>(&vertices[i]);
      for (auto a : VboHelpers::kOrderedAttrs) {
        const auto a_sz = VboHelpers::size(a);
        if (VboHelpers::has_attr(config_.vbo_attrs, a)) {
          memcpy(mapped_vbo, src, a_sz);
          mapped_vbo = (uint8_t *)mapped_vbo + a_sz;
        }
        // Pointer to the full Vertex attributes needs to be advanced
        // unconditionally.
        src += a_sz;
      }
    }
    app_context_->device().unmap(staging_vertex_buffer_);
  }

  app_context_->device().memcpy_internal(
      vertex_buffer_.get_ptr(0), staging_vertex_buffer_.get_ptr(0),
      config_.vertices_count * config_.vbo_size());
}

void SetImage::update_index_buffer() {
  const std::vector<uint32_t> indices = {
      0, 1, 2, 3, 4, 5,
  };
  {
    void *mapped_ibo{nullptr};
    TI_ASSERT(app_context_->device().map(staging_index_buffer_, &mapped_ibo) ==
              RhiResult::success);
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

}  // namespace taichi::ui
