#include "taichi/rhi/metal/metal_command_list.h"
#include "taichi/rhi/metal/metal_device.h"
#include "taichi/rhi/metal/metal_pipeline.h"
#include "taichi/rhi/metal/metal_stream.h"
#include "taichi/rhi/metal/metal_resource_binder.h"

namespace taichi::lang::metal {

MetalCommandList::MetalCommandList(MetalStream* stream,
                  mac::nsobj_unique_ptr<MTL::CommandBuffer>&& command_buffer) :
    stream_(stream), command_buffer_(std::move(command_buffer)) {
}

std::unique_ptr<MetalCommandList> MetalCommandList::create(MetalStream* stream) {
  // (penguinliong) `commandBuffer` are allocated in pool. But to fit our Device
  // API design we have to decouple the lifetime of the command buffer and the
  // pool.
  mac::ScopedAutoreleasePool autorelease;
  mac::nsobj_unique_ptr<MTL::CommandBuffer> command_buffer =
      mac::retain_and_wrap_as_nsobj_unique_ptr(
        stream->get_mtl_command_queue()->commandBuffer());
  TI_ASSERT(command_buffer != nullptr);

  auto label = fmt::format("command_buffer_{}", (size_t)command_buffer.get());
  command_buffer->setLabel(mac::wrap_string_as_ns_string(label).get());

  return std::make_unique<MetalCommandList>(stream, std::move(command_buffer));
}

void MetalCommandList::bind_pipeline(Pipeline *pipeline) {
  current_bound_pipeline_ = static_cast<MetalPipeline *>(pipeline);
}
void MetalCommandList::bind_resources(ResourceBinder *binder) {
  current_bound_resource_binder_ = static_cast<MetalResourceBinder *>(binder);
}

void MetalCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  TI_ERROR_IF(dst.device != src.device,
              "dst and src must be from the same MTLDevice");
  MTL::Buffer *dst_buf = stream_->get_device()->get_mtl_buffer(dst.alloc_id);
  TI_ASSERT(dst_buf != nullptr);
  MTL::Buffer *src_buf = stream_->get_device()->get_mtl_buffer(src.alloc_id);
  TI_ASSERT(src_buf != nullptr);

  mac::ScopedAutoreleasePool autorelease;
  MTL::BlitCommandEncoder *encoder = command_buffer_->blitCommandEncoder();
  TI_ASSERT(encoder != nullptr);
  encoder->copyFromBuffer(src_buf, src.offset, dst_buf, dst.offset, size);
  encoder->endEncoding();
}
void MetalCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  if ((data & 0xff) != data) {
    // TODO: Maybe create a shader just for this filling purpose?
    TI_ERROR("Metal can only support 8-bit data for buffer_fill");
    return;
  }
  MTL::Buffer *buf = stream_->get_device()->get_mtl_buffer(ptr.alloc_id);
  TI_ASSERT(buf != nullptr);

  mac::ScopedAutoreleasePool autorelease;
  MTL::BlitCommandEncoder *encoder = command_buffer_->blitCommandEncoder();
  TI_ASSERT(encoder != nullptr);
  encoder->fillBuffer(buf, NS::Range(ptr.offset, size), (data & 0xff));
  encoder->endEncoding();
}

void MetalCommandList::dispatch(CommandList::ComputeSize grid_size,
              CommandList::ComputeSize block_size) {
  mac::ScopedAutoreleasePool autorelease;
  MTL::ComputeCommandEncoder *encoder =
      command_buffer_->computeCommandEncoder();
  TI_ASSERT(encoder != nullptr);

  // Bind compute pipeline.
  TI_ASSERT(current_bound_pipeline_ != nullptr);
  encoder->setComputePipelineState(current_bound_pipeline_->get_mtl_pipeline_state());

  // Bind resources.
  for (const auto& pair : current_bound_resource_binder_->get_bindings()) {
    encoder->setBuffer(pair.second.buffer, pair.second.offset, pair.first);
  }

  auto ceil_div = [](uint32_t a, uint32_t b) -> uint32_t {
    return (a + b - 1) / b;
  };
  MTL::Size block_count(ceil_div(grid_size.x, block_size.x),
                        ceil_div(grid_size.y, block_size.y),
                        ceil_div(grid_size.z, block_size.z));
  MTL::Size block_size2(block_size.x, block_size.y, block_size.z);
  encoder->dispatchThreadgroups(block_count, block_size2);
  encoder->endEncoding();
}

} // namespace taichi::lang::metal
