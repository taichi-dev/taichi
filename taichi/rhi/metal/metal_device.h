#pragma once
#include <memory>
#include "taichi/common/logging.h"
#include "taichi/rhi/device.h"
#include "taichi/runtime/metal/api.h"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <CoreGraphics/CoreGraphics.h>
#define DEFINE_METAL_ID_TYPE(x) typedef id<x> x##_id;
#else
#define DEFINE_METAL_ID_TYPE(x) typedef struct x##_t *x##_id;
#endif

DEFINE_METAL_ID_TYPE(MTLDevice);
DEFINE_METAL_ID_TYPE(MTLBuffer);
DEFINE_METAL_ID_TYPE(MTLLibrary);
DEFINE_METAL_ID_TYPE(MTLFunction);
DEFINE_METAL_ID_TYPE(MTLComputePipelineState);
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

struct MetalWorkgroupSize {
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};
};
class MetalPipeline : public Pipeline {
 public:
  explicit MetalPipeline(const MetalDevice &device,
                         MTLLibrary_id mtl_library,
                         MTLFunction_id mtl_function,
                         MTLComputePipelineState_id mtl_compute_pipeline_state,
                         MetalWorkgroupSize workgroup_size);
  ~MetalPipeline() override;

  static MetalPipeline* create(const MetalDevice &device,
                                               const uint32_t *spv_data,
                                               size_t spv_size);
  void destroy();

  inline MTLComputePipelineState_id mtl_compute_pipeline_state() const {
    return mtl_compute_pipeline_state_;
  }
  inline const MetalWorkgroupSize &workgroup_size() const {
    return workgroup_size_;
  }

 private:
  const MetalDevice *device_;
  MTLLibrary_id mtl_library_;
  MTLFunction_id mtl_function_;
  MTLComputePipelineState_id mtl_compute_pipeline_state_;
  MetalWorkgroupSize workgroup_size_;
  bool is_destroyed_{false};
};

enum class MetalShaderResourceType {
  buffer,
};
struct MetalShaderBufferResource {
  MTLBuffer_id buffer;
  size_t offset;
  size_t size;
};
struct MetalShaderImageResource {
  // TODO: (penguinliong)
};
struct MetalShaderResource {
  MetalShaderResourceType ty;
  uint32_t binding;
  union {
    MetalShaderBufferResource buffer;
    MetalShaderImageResource image;
  };
};
class MetalShaderResourceSet : public ShaderResourceSet {
 public:
  explicit MetalShaderResourceSet(const MetalDevice &device);
  ~MetalShaderResourceSet() override;

  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DevicePtr ptr,
                               size_t size) override;
  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DeviceAllocation alloc) override;

  ShaderResourceSet &buffer(uint32_t binding,
                            DevicePtr ptr,
                            size_t size) override;
  ShaderResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) override;

  inline const std::vector<MetalShaderResource> &resources() const {
    return resources_;
  }

 private:
  const MetalDevice *device_;
  std::vector<MetalShaderResource> resources_;
};

class MetalCommandList : public CommandList {
 public:
  // `mtl_command_buffer` should be already retained.
  explicit MetalCommandList(const MetalDevice &device);
  ~MetalCommandList() override;

  void bind_pipeline(Pipeline *p) override;
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
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;

 private:
  friend class MetalStream;

  const MetalDevice *device_;
  std::vector<std::function<void(MTLCommandBuffer_id)>> pending_commands_;

  // Non-null after `bind*` methods.
  const MetalPipeline *current_pipeline_;
  const MetalShaderResourceSet *current_shader_resource_set_;
};

class MetalStream : public Stream {
 public:
  // `mtl_command_queue` should be already retained.
  explicit MetalStream(const MetalDevice &device,
                       MTLCommandQueue_id mtl_command_queue);
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
  const MetalDevice *device_;
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

  std::unique_ptr<Pipeline> create_pipeline(const PipelineSourceDesc &src,
                                            std::string name) override;
  ShaderResourceSet *create_resource_set() override;

  Stream *get_compute_stream() override;
  void wait_idle() override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

 private:
  MTLDevice_id mtl_device_;
  std::map<DeviceAllocationId, std::unique_ptr<MetalMemory>> memory_allocs_;
  std::unique_ptr<MetalStream> compute_stream_;

  bool is_destroyed_{false};
};

}  // namespace metal
}  // namespace taichi::lang
