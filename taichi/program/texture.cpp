#include "taichi/program/texture.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

Texture::Texture(Program *prog,
                 BufferFormat format,
                 int width,
                 int height,
                 int depth)
    : format_(format),
      width_(width),
      height_(height),
      depth_(depth),
      prog_(prog) {
  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  auto [type, num_channels] = buffer_format2type_channels(format);
  TI_TRACE("Create image, gfx device {}, format={}, w={}, h={}, d={}",
           (void *)device, type.to_string(), num_channels, width, height,
           depth);

  TI_ASSERT(num_channels > 0 && num_channels <= 4);

  ImageParams img_params{};
  img_params.dimension = depth > 1 ? ImageDimension::d3D : ImageDimension::d2D;
  img_params.format = format;
  img_params.x = width;
  img_params.y = height;
  img_params.z = depth;
  img_params.initial_layout = ImageLayout::undefined;
  texture_alloc_ = prog_->allocate_texture(img_params);

  format_ = img_params.format;

  TI_TRACE("image created, gfx device {}", (void *)device);
}

Texture::Texture(DeviceAllocation &devalloc,
                 BufferFormat format,
                 int width,
                 int height,
                 int depth)
    : texture_alloc_(devalloc),
      format_(format),
      width_(width),
      height_(height),
      depth_(depth) {
  format_ = format;
}

intptr_t Texture::get_device_allocation_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(&texture_alloc_);
}

void Texture::from_ndarray(Ndarray *ndarray) {
  auto semaphore = prog_->flush();

  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  device->image_transition(texture_alloc_, ImageLayout::undefined,
                           ImageLayout::transfer_dst);

  Stream *stream = device->get_compute_stream();
  auto cmdlist = stream->new_command_list();

  BufferImageCopyParams params;
  params.buffer_row_length = ndarray->shape[0];
  params.buffer_image_height = ndarray->shape[1];
  params.image_mip_level = 0;
  params.image_extent.x = width_;
  params.image_extent.y = height_;
  params.image_extent.z = depth_;

  cmdlist->buffer_barrier(ndarray->ndarray_alloc_);
  cmdlist->buffer_to_image(texture_alloc_, ndarray->ndarray_alloc_.get_ptr(0),
                           ImageLayout::transfer_dst, params);

  stream->submit_synced(cmdlist.get(), {semaphore});
}

DevicePtr get_device_ptr(taichi::lang::Program *program, SNode *snode) {
  SNode *dense_parent = snode->parent;
  SNode *root = dense_parent->parent;

  int tree_id = root->get_snode_tree_id();
  DevicePtr root_ptr = program->get_snode_tree_device_ptr(tree_id);

  return root_ptr.get_ptr(program->get_field_in_tree_offset(tree_id, snode));
}

void Texture::from_snode(SNode *snode) {
  auto semaphore = prog_->flush();

  TI_ASSERT(snode->is_path_all_dense);

  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  device->image_transition(texture_alloc_, ImageLayout::undefined,
                           ImageLayout::transfer_dst);

  DevicePtr devptr = get_device_ptr(prog_, snode);

  Stream *stream = device->get_compute_stream();
  auto cmdlist = stream->new_command_list();

  BufferImageCopyParams params;
  params.buffer_row_length = snode->shape_along_axis(0);
  params.buffer_image_height = snode->shape_along_axis(1);
  params.image_mip_level = 0;
  params.image_extent.x = width_;
  params.image_extent.y = height_;
  params.image_extent.z = depth_;

  cmdlist->buffer_barrier(devptr);
  cmdlist->buffer_to_image(texture_alloc_, devptr, ImageLayout::transfer_dst,
                           params);

  stream->submit_synced(cmdlist.get(), {semaphore});
}

Texture::~Texture() {
  if (prog_) {
    GraphicsDevice *device =
        static_cast<GraphicsDevice *>(prog_->get_graphics_device());
    device->destroy_image(texture_alloc_);
  }
}

}  // namespace lang
}  // namespace taichi
