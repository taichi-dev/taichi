#pragma once
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"

namespace taichi::lang::metal {

struct MetalCommandList : public CommandList {
public:
  explicit MetalCommandList(MetalStream* stream,
                   mac::nsobj_unique_ptr<MTL::CommandBuffer>&& command_buffer);

  static std::unique_ptr<MetalCommandList> create(MetalStream* stream);

  void bind_pipeline(Pipeline *pipeline) override;
  void bind_resources(ResourceBinder *binder) override;
  void bind_resources(ResourceBinder *binder,
                      ResourceBinder::Bindings *bindings) override {
    TI_NOT_IMPLEMENTED;
  }

  void buffer_barrier(DevicePtr ptr, size_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void buffer_barrier(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }
  void memory_barrier() override {
    TI_NOT_IMPLEMENTED;
  }

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override;

  void dispatch(uint32_t x, uint32_t y, uint32_t z) override {
    TI_ERROR("Please call dispatch(grid_size, block_size) instead");
  }
  void dispatch(CommandList::ComputeSize grid_size,
                CommandList::ComputeSize block_size) override;

  constexpr MTL::CommandBuffer *get_mtl_command_buffer() const {
    return command_buffer_.get();
  }

private:
  MetalStream *stream_;
  mac::nsobj_unique_ptr<MTL::CommandBuffer> command_buffer_;

  MetalPipeline *current_bound_pipeline_;
  MetalResourceBinder *current_bound_resource_binder_;
};

} // taichi::lang::metal
