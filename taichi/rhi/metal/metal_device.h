#pragma once
#include <memory>
#include "taichi/common/logging.h"
#include "taichi/rhi/device.h"
#include "taichi/runtime/metal/api.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#define DEFINE_METAL_ID_TYPE(x) typedef id<x> x##_id;
#else
#define DEFINE_METAL_ID_TYPE(x) typedef struct x##_t *x##_id;
#endif

DEFINE_METAL_ID_TYPE(MTLDevice);
DEFINE_METAL_ID_TYPE(MTLBuffer);
DEFINE_METAL_ID_TYPE(MTLCommandQueue);
DEFINE_METAL_ID_TYPE(MTLCommandBuffer);
DEFINE_METAL_ID_TYPE(MTLBlitCommandEncoder);
DEFINE_METAL_ID_TYPE(MTLComputeCommandEncoder);

#undef DEFINE_METAL_ID_TYPE

namespace taichi::lang {

namespace metal {

struct MetalMemory;
class MetalCommandList;
class MetalStream;
class MetalDevice;

struct MetalMemory {
 public:
  // `mtl_buffer` should be already retained.
  explicit MetalMemory(MTLBuffer_id mtl_buffer);
  ~MetalMemory();

  MTLBuffer_id mtl_buffer() const;
  size_t size() const;
  RhiResult mapped_ptr(void **mapped_ptr) const;

 private:
  MTLBuffer_id mtl_buffer_;
};


class MetalCommandList : public CommandList {
 public:
  // `mtl_command_buffer` should be already retained.
  explicit MetalCommandList(const MetalDevice &device);
  ~MetalCommandList() override;

  void bind_pipeline(Pipeline *p) override {
    TI_NOT_IMPLEMENTED
  }
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept override {
    TI_NOT_IMPLEMENTED
  }
  
  void buffer_barrier(DevicePtr ptr, size_t size) override {
    TI_NOT_IMPLEMENTED
  }
  void buffer_barrier(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED
  }
  void memory_barrier() override;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override;
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override {
    TI_NOT_IMPLEMENTED
  }

 private:
  friend class MetalStream;

  const MetalDevice *device_;
  std::vector<std::function<void(MTLCommandBuffer_id)>> pending_commands_;
};


class MetalStream : public Stream {
 public:
  // `mtl_command_queue` should be already retained.
  explicit MetalStream(const MetalDevice& device, MTLCommandQueue_id mtl_command_queue);
  ~MetalStream() override;

  MTLCommandQueue_id mtl_command_queue() const {
    return mtl_command_queue_;
  }

  std::unique_ptr<CommandList> new_command_list() override;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void command_sync() override;

 private:
  const MetalDevice* device_;
  MTLCommandQueue_id mtl_command_queue_;
  std::vector<MTLCommandBuffer_id> pending_cmdbufs_;
};


class MetalDevice : public Device {
 public:
  // `mtl_device` should be already retained.
  explicit MetalDevice(MTLDevice_id mtl_device);
  ~MetalDevice() override;

  Arch arch() const override {
    return Arch::metal;
  }
  MTLDevice_id mtl_device() const {
    return mtl_device_;
  }

  static std::unique_ptr<MetalDevice> create();
  void destroy();

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;
  const MetalMemory &get_memory(DeviceAllocationId alloc_id) const;

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) override;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) override;
  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation ptr) override;

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name) override;

  Stream *get_compute_stream() override;
  void wait_idle() override;

  ShaderResourceSet *create_resource_set() override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

 private:
  MTLDevice_id mtl_device_;
  std::map<DeviceAllocationId, std::unique_ptr<MetalMemory>> memory_allocs_;
  std::unique_ptr<MetalStream> compute_stream_;

  bool is_destroyed_{false};
};

}  // namespace metal
}  // namespace taichi::lang
