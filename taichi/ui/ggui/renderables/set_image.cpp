#include "set_image.h"

#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void SetImage::update_ubo(float x_factor, float y_factor, bool transpose) {
  glm::vec2 pixel_size = glm::vec2(1.0f / width_, 1.0f / height_);
  glm::vec2 lower_bound = pixel_size * 0.5f;
  glm::vec2 upper_bound = glm::vec2(1.0f, 1.0f) - pixel_size * 0.5f;
  UniformBufferObject ubo = {lower_bound, upper_bound, x_factor, y_factor,
                             int(transpose)};
  void *mapped{nullptr};
  RHI_VERIFY(app_context_->device().map(uniform_buffer_renderable_->get_ptr(0),
                                        &mapped));
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(*uniform_buffer_renderable_);
}

void SetImage::update_data(const SetImageInfo &info) {
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  StreamSemaphore sema = nullptr;

  const FieldInfo &img = info.img;

  // Image is a width x height field of u32 which contains encoded RGBA8
  TI_ASSERT_INFO(
      img.shape.size() == 2 && img.dtype == taichi::lang::PrimitiveType::u32,
      "set_image buffer input must be 2D field of u32");

  int new_width = img.shape[0];
  int new_height = img.shape[1];

  resize_texture(new_width, new_height, BufferFormat::rgba8);

  update_ubo(1.0f, 1.0f, true);

  const uint64_t img_size_bytes = width_ * height_ * sizeof(uint32_t);

  // If data source is not a host mapped pointer, it is a DeviceAllocation
  // from the same backend as GGUI
  DevicePtr img_dev_ptr = info.img.dev_alloc.get_ptr();
  bool uses_host = img.field_source == FieldSource::HostMappedPtr;
  if (uses_host) {
    DeviceAllocation staging;
    RhiResult res = app_context_->device().allocate_memory(
        {img_size_bytes, true, false, false, AllocUsage::None}, &staging);
    TI_ASSERT(res == RhiResult::success);

    // Map the staing buffer and perform memcpy
    void *dst_ptr{nullptr};
    RHI_VERIFY(app_context_->device().map(staging.get_ptr(), &dst_ptr));
    void *src_ptr = reinterpret_cast<uint8_t *>(img.dev_alloc.alloc_id);
    memcpy(dst_ptr, src_ptr, img_size_bytes);
    app_context_->device().unmap(staging);

    img_dev_ptr = staging.get_ptr(0);
  }

  auto copy_op = [&, img_dev_ptr, uses_host](Device *device,
                                             CommandList *cmdlist) {
    BufferImageCopyParams copy_params;
    // these are flipped because taichi is y-major and vulkan is x-major
    copy_params.image_extent.x = height_;
    copy_params.image_extent.y = width_;
    cmdlist->image_transition(*texture_, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->buffer_barrier(img_dev_ptr);
    cmdlist->buffer_to_image(*texture_, img_dev_ptr, ImageLayout::transfer_dst,
                             copy_params);
    cmdlist->image_transition(*texture_, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
    if (uses_host) {
      device->dealloc_memory(img_dev_ptr);
    }
  };

  if (prog && prog->get_graphics_device() == &app_context_->device()) {
    // If it's the same device, we do not use the staging buffer and directly
    // copy from the src ptr to the image
    prog->enqueue_compute_op_lambda(copy_op, {});
  } else {
    // Create a single time command
    auto stream = app_context_->device().get_graphics_stream();
    auto [cmdlist, res] = stream->new_command_list_unique();
    TI_ASSERT_INFO(res == RhiResult::success,
                   "Failed to allocate command list");
    copy_op(&app_context_->device(), cmdlist.get());
    if (sema) {
      stream->submit(cmdlist.get(), {sema});
    } else {
      stream->submit(cmdlist.get());
    }
  }
}

void SetImage::update_data(Texture *tex) {
  Program *prog = app_context_->prog();

  auto shape = tex->get_size();
  auto new_format = tex->get_buffer_format();

  TI_ASSERT_INFO(shape[2] == 1,
                 "Must be a 2D image! Received image shape: {}x{}x{}", shape[0],
                 shape[1], shape[2]);

  // Reminder: y/x is flipped in Taichi. I would like to use the correct
  // orientation, but we have existing code already using the previous
  // convention
  const int new_width = shape[1];
  const int new_height = shape[0];
  resize_texture(new_width, new_height, new_format);

  update_ubo(1.0f, 1.0f, false);

  ImageCopyParams copy_params;
  copy_params.width = shape[0];
  copy_params.height = shape[1];
  copy_params.depth = shape[2];

  DeviceAllocation src_alloc = tex->get_device_allocation();
  auto copy_op = [&, src_alloc](Device *device, CommandList *cmdlist) {
    cmdlist->image_transition(*this->texture_, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->copy_image(*this->texture_, src_alloc, ImageLayout::transfer_dst,
                        ImageLayout::transfer_src, copy_params);
    cmdlist->image_transition(*this->texture_, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
  };

  // In the current state if we called this direct image update data method, we
  // gurantee to have a program.
  // FIXME: However, if we don't have a Program, where does the layout come
  // from?
  if (prog && prog->get_graphics_device() == &app_context_->device()) {
    prog->enqueue_compute_op_lambda(
        copy_op, {ComputeOpImageRef{src_alloc, ImageLayout::transfer_src,
                                    ImageLayout::transfer_src}});
  } else {
    TI_ERROR("`update_data` received Texture from a different device");
  }
}

SetImage::SetImage(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.draw_vertex_count = 6;
  config.ubo_size = sizeof(UniformBufferObject);
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/SetImage_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/SetImage_vk_vert.spv";

  Renderable::init(config, app_context);
  create_graphics_pipeline();

  // Create UBO
  {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {config_.ubo_size, /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Uniform});
    TI_ASSERT(res == RhiResult::success);
    uniform_buffer_renderable_ = std::move(buf);
  }

  // Create & upload vertex buffer (constant)
  const std::vector<Vertex> vertices = {
      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{-1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},

      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 1.f, 1.f}},
  };
  {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {sizeof(Vertex) * vertices.size(), /*host_write=*/true,
         /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Vertex});
    TI_ASSERT(res == RhiResult::success);
    vertex_buffer_ = std::move(buf);
  }
  void *mapped_vbo{nullptr};
  RHI_VERIFY(
      app_context_->device().map(vertex_buffer_->get_ptr(0), &mapped_vbo));
  memcpy(mapped_vbo, vertices.data(), sizeof(Vertex) * vertices.size());
  app_context_->device().unmap(*vertex_buffer_);
}

void SetImage::record_this_frame_commands(CommandList *command_list) {
  resource_set_->image(0, *texture_, {});
  resource_set_->buffer(1, uniform_buffer_renderable_->get_ptr());

  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vertex_buffer_->get_ptr(), 0);

  command_list->bind_pipeline(pipeline_);
  command_list->bind_raster_resources(raster_state.get());
  command_list->bind_shader_resources(resource_set_.get());
  command_list->draw(6);
}

void SetImage::resize_texture(int width,
                              int height,
                              taichi::lang::BufferFormat format) {
  if (width_ == width && height_ == height && format_ == format &&
      texture_ != nullptr) {
    return;
  }

  texture_.reset();

  width_ = width;
  height_ = height;
  format_ = format;

  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = format_;
  params.initial_layout = ImageLayout::undefined;
  // these are flipped because taichi is y-major and vulkan is x-major
  params.x = height_;
  params.y = width_;
  params.z = 1;
  params.export_sharing = false;

  texture_ = app_context_->device().create_image_unique(params);
}

}  // namespace vulkan

}  // namespace taichi::ui
