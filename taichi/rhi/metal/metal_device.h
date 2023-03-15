#pragma once
#include <memory>
#include "taichi/common/logging.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"
#include "taichi/rhi/impl_support.h"

#if defined(__APPLE__) && defined(__OBJC__)
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
DEFINE_METAL_ID_TYPE(MTLTexture);
DEFINE_METAL_ID_TYPE(MTLSamplerState);
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
struct MetalImage;
struct MetalSampler;
class MetalCommandList;
class MetalStream;
class MetalDevice;

struct MetalMemory {
 public:
  // `mtl_buffer` should be already retained.
  explicit MetalMemory(MTLBuffer_id mtl_buffer, bool host_access);
  ~MetalMemory();

  void dont_destroy();

  MTLBuffer_id mtl_buffer() const;
  size_t size() const;
  RhiResult mapped_ptr(void **mapped_ptr) const;

 private:
  MTLBuffer_id mtl_buffer_;
  bool can_map_{false};
  bool dont_destroy_{false};
};

struct MetalImage {
 public:
  // `mtl_texture` should be already retained.
  explicit MetalImage(MTLTexture_id mtl_texture);
  ~MetalImage();

  void dont_destroy();

  MTLTexture_id mtl_texture() const;

 private:
  MTLTexture_id mtl_texture_;
  bool dont_destroy_{false};
};

struct MetalSampler {
 public:
  // `mtl_texture` should be already retained.
  explicit MetalSampler(MTLSamplerState_id mtl_sampler_state);
  ~MetalSampler();

  MTLSamplerState_id mtl_sampler_state() const;

 private:
  MTLSamplerState_id mtl_sampler_state_;
};

struct MetalWorkgroupSize {
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};
};
class MetalPipeline final : public Pipeline {
 public:
  // `mtl_library`, `mtl_function`, `mtl_compute_pipeline_state` should be
  // already retained.
  explicit MetalPipeline(const MetalDevice &device,
                         MTLLibrary_id mtl_library,
                         MTLFunction_id mtl_function,
                         MTLComputePipelineState_id mtl_compute_pipeline_state,
                         MetalWorkgroupSize workgroup_size);
  ~MetalPipeline() final;

  static MetalPipeline *create(const MetalDevice &device,
                               const uint32_t *spv_data,
                               size_t spv_size,
                               const std::string &name);
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
  texture,
};
struct MetalShaderBufferResource {
  MTLBuffer_id buffer;
  size_t offset;
  size_t size;
};
struct MetalShaderTextureResource {
  MTLTexture_id texture;
  bool is_sampled;
};
struct MetalShaderResource {
  MetalShaderResourceType ty;
  uint32_t binding;
  union {
    MetalShaderBufferResource buffer;
    MetalShaderTextureResource texture;
  };
};
class MetalShaderResourceSet final : public ShaderResourceSet {
 public:
  explicit MetalShaderResourceSet(const MetalDevice &device);
  ~MetalShaderResourceSet() final;

  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DevicePtr ptr,
                               size_t size) final;
  ShaderResourceSet &rw_buffer(uint32_t binding, DeviceAllocation alloc) final;

  ShaderResourceSet &buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  ShaderResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) final;

  ShaderResourceSet &image(uint32_t binding,
                           DeviceAllocation alloc,
                           ImageSamplerConfig sampler_config) override;

  ShaderResourceSet &rw_image(uint32_t binding,
                              DeviceAllocation alloc,
                              int lod) override;

  inline const std::vector<MetalShaderResource> &resources() const {
    return resources_;
  }

 private:
  const MetalDevice *device_;
  std::vector<MetalShaderResource> resources_;
};

class MetalCommandList final : public CommandList {
 public:
  explicit MetalCommandList(const MetalDevice &device,
                            MTLCommandQueue_id cmd_queue);
  ~MetalCommandList() final;

  void bind_pipeline(Pipeline *p) noexcept final;
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final;
  RhiResult bind_raster_resources(RasterResources *res) noexcept final;

  void buffer_barrier(DevicePtr ptr, size_t size) noexcept final;
  void buffer_barrier(DeviceAllocation alloc) noexcept final;
  void memory_barrier() noexcept final;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final;
  RhiResult dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept final;

  void image_transition(DeviceAllocation img,
                        ImageLayout old_layout,
                        ImageLayout new_layout) final;

  MTLCommandBuffer_id finalize();

 private:
  friend class MetalStream;

  const MetalDevice *device_;
  MTLCommandBuffer_id cmdbuf_;

  // Non-null after `bind*` methods.
  const MetalPipeline *current_pipeline_;
  const MetalShaderResourceSet *current_shader_resource_set_;
};

class MetalStream final : public Stream {
 public:
  // `mtl_command_queue` should be already retained.
  explicit MetalStream(const MetalDevice &device,
                       MTLCommandQueue_id mtl_command_queue);
  ~MetalStream() override;

  static MetalStream *create(const MetalDevice &device);
  void destroy();

  MTLCommandQueue_id mtl_command_queue() const {
    return mtl_command_queue_;
  }

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) final;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) final;

  void command_sync() override;

 private:
  const MetalDevice *device_;
  MTLCommandQueue_id mtl_command_queue_;
  std::vector<MTLCommandBuffer_id> pending_cmdbufs_;
  bool is_destroyed_{false};
};

class MetalDevice final : public GraphicsDevice {
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

  static MetalDevice *create();
  void destroy();

  std::unique_ptr<Surface> create_surface(
      const SurfaceConfig &config) override {
    TI_NOT_IMPLEMENTED;
  }

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  DeviceAllocation import_mtl_buffer(MTLBuffer_id buffer);
  void dealloc_memory(DeviceAllocation handle) override;

  DeviceAllocation create_image(const ImageParams &params) override;
  DeviceAllocation import_mtl_texture(MTLTexture_id texture);
  void destroy_image(DeviceAllocation handle) override;

  const MetalMemory &get_memory(DeviceAllocationId alloc_id) const;
  MetalMemory &get_memory(DeviceAllocationId alloc_id);

  const MetalImage &get_image(DeviceAllocationId alloc_id) const;
  MetalImage &get_image(DeviceAllocationId alloc_id);

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) override;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) override;
  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation ptr) override;

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final;
  ShaderResourceSet *create_resource_set() override;

  std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override {
    TI_NOT_IMPLEMENTED;
  }
  RasterResources *create_raster_resources() override {
    TI_NOT_IMPLEMENTED;
  }

  Stream *get_compute_stream() override;
  Stream *get_graphics_stream() override;
  void wait_idle() override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  const MetalSampler &get_default_sampler() const {
    return *default_sampler_;
  }

 private:
  MTLDevice_id mtl_device_;
  rhi_impl::SyncedPtrStableObjectList<MetalMemory> memory_allocs_;
  rhi_impl::SyncedPtrStableObjectList<MetalImage> image_allocs_;
  std::unique_ptr<MetalStream> compute_stream_;
  std::unique_ptr<MetalStream> graphics_stream_;
  std::unique_ptr<MetalSampler> default_sampler_;

  bool is_destroyed_{false};
};

}  // namespace metal
}  // namespace taichi::lang
