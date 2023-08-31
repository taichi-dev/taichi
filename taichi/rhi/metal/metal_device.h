#pragma once
#include "taichi/common/logging.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/impl_support.h"
#include "taichi/rhi/metal/metal_api.h"
#include <memory>
#include <regex>

// clang-format off
#if defined(__APPLE__) && defined(__OBJC__)
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#define DEFINE_METAL_ID_TYPE(x) typedef id<x> x##_id;
#define DEFINE_OBJC_TYPE(x) // Should be defined by included headers
#else
#define DEFINE_METAL_ID_TYPE(x) typedef struct x##_t *x##_id;
#define DEFINE_OBJC_TYPE(x) typedef void x;
#endif
// clang-format on

DEFINE_METAL_ID_TYPE(MTLDevice);
DEFINE_METAL_ID_TYPE(MTLBuffer);
DEFINE_METAL_ID_TYPE(MTLTexture);
DEFINE_METAL_ID_TYPE(MTLSamplerState);
DEFINE_METAL_ID_TYPE(MTLLibrary);
DEFINE_METAL_ID_TYPE(MTLFunction);
DEFINE_METAL_ID_TYPE(MTLComputePipelineState);
DEFINE_METAL_ID_TYPE(MTLRenderPipelineState);
DEFINE_METAL_ID_TYPE(MTLCommandQueue);
DEFINE_METAL_ID_TYPE(MTLCommandBuffer);
DEFINE_METAL_ID_TYPE(MTLBlitCommandEncoder);
DEFINE_METAL_ID_TYPE(MTLComputeCommandEncoder);
DEFINE_METAL_ID_TYPE(MTLRenderCommandEncoder);
DEFINE_METAL_ID_TYPE(CAMetalDrawable);
DEFINE_METAL_ID_TYPE(MTLDepthStencilState);
DEFINE_OBJC_TYPE(CAMetalLayer);
DEFINE_OBJC_TYPE(MTLVertexDescriptor);
DEFINE_OBJC_TYPE(MTLRenderPassDescriptor);

#undef DEFINE_METAL_ID_TYPE
#undef DEFINE_OBJC_TYPE

/*
TODO LIST:
  - Support multiple ShaderResourceSet's bounded in a renderpass.
    - Use argument buffers? Maybe not necessary. Can just use current system and
      store a list of resource sets, and bind all of the data rather than using
      bindless.
  - Optimize draw call RenderCommandEncoder usage.
    - Each draw call creates its own RenderCommandEncoder. This is inefficient
      since ending encoding comes with significant cost. Best to try and keep
      the encoders alive for as long as possible, and when a non-render command
      is called, end the current RenderEncoder's encoding and the next draw call
      will create a new one.
    - This has implications in the ImGUI part of GGUI though, since that
      requires a render encoder to be passed to ImGUI. Right now it just creates
      a new render encoder for each ImGUI draw call too, but with this
      optimization it would get the current render encoder or create a new one
      if there is none.
*/

namespace taichi::lang {

namespace metal {

struct MetalMemory;
struct MetalImage;
struct MetalSampler;
class MetalCommandList;
class MetalStream;
class MetalDevice;

struct MetalMemory : public rhi_impl::NonAssignable {
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

struct MetalImage : public rhi_impl::NonAssignable {
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

struct MetalSampler : public rhi_impl::NonAssignable {
 public:
  // `mtl_texture` should be already retained.
  explicit MetalSampler(MTLSamplerState_id mtl_sampler_state);
  ~MetalSampler();

  // No copy constructor
  MetalSampler(MetalSampler &other) = delete;

  MTLSamplerState_id mtl_sampler_state() const;

 private:
  MTLSamplerState_id mtl_sampler_state_;
};

struct MetalRenderPassTargetDetails {
  std::vector<std::pair<BufferFormat, bool>> color_attachments;
  BufferFormat depth_attach_format{BufferFormat::unknown};
  bool clear_depth{false};

  bool operator==(const MetalRenderPassTargetDetails &other) const {
    if (other.depth_attach_format != depth_attach_format) {
      return false;
    }
    if (other.clear_depth != clear_depth) {
      return false;
    }
    return other.color_attachments == color_attachments;
  }
};
struct MRPTDHasher {
  std::size_t operator()(const MetalRenderPassTargetDetails &desc) const {
    size_t hash = std::hash<uint64_t>()(
        (uint64_t(desc.depth_attach_format) << 1) | uint64_t(desc.clear_depth));
    for (auto &pair : desc.color_attachments) {
      size_t hash_pair = std::hash<uint64_t>()((uint64_t(pair.first) << 1) |
                                               uint64_t(pair.second));
      rhi_impl::hash_combine(hash, hash_pair);
    }
    return hash;
  }
};
struct MetalWorkgroupSize {
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};
};

struct MetalRasterLibraries {
  MetalRasterLibraries();

  MTLLibrary_id vertex;
  MTLLibrary_id fragment;

  void destroy();
};

struct MetalRasterFunctions {
  MetalRasterFunctions();

  MTLFunction_id vertex;
  MTLFunction_id fragment;

  void destroy();
};

struct MetalShaderBindingMapping {
  // Map GLSL binding to MSL index (buffer/texture index, sampler index)
  std::unordered_map<int, std::pair<int, int>> vertex;
  std::unordered_map<int, std::pair<int, int>> fragment;
  // The highest used buffer index used in the MSL vertex function
  // Vertex attributes use up vertex buffer indices in Metal, but
  // SPIRV-cross generated MSL doesn't know how many buffer indices
  // the vertex attributes will take. So bind vertex input after the other
  // buffers.
  int max_vert_buffer_index{-1};
};

class MetalPipeline final : public Pipeline, public rhi_impl::NonAssignable {
 public:
  // `mtl_library`, `mtl_function`, `mtl_compute_pipeline_state` should be
  // already retained.
  explicit MetalPipeline(const MetalDevice &device,
                         MTLLibrary_id mtl_library,
                         MTLFunction_id mtl_function,
                         MTLComputePipelineState_id mtl_compute_pipeline_state,
                         MetalWorkgroupSize workgroup_size);

  explicit MetalPipeline(const MetalDevice &device,
                         MetalRasterLibraries &mtl_libraries,
                         MetalRasterFunctions &mtl_functions,
                         MTLVertexDescriptor *vertex_descriptor,
                         const MetalShaderBindingMapping &mapping,
                         const RasterParams &raster_params);
  ~MetalPipeline() final;

  static MetalPipeline *create_compute_pipeline(const MetalDevice &device,
                                                const uint32_t *spv_data,
                                                size_t spv_size,
                                                const std::string &name);

  MTLRenderPipelineState_id build_mtl_render_pipeline(
      const MetalRenderPassTargetDetails &renderpass_details);

  inline MTLComputePipelineState_id mtl_compute_pipeline_state() const {
    return mtl_compute_pipeline_state_;
  }
  inline const MetalWorkgroupSize &workgroup_size() const {
    return workgroup_size_;
  }

  const RasterParams *raster_params() const {
    return &raster_params_;
  }
  const MetalShaderBindingMapping *bind_map() const {
    return &binding_mapping_;
  }

  bool is_graphics() const {
    return is_raster_pipeline_;
  }

  std::unordered_map<MetalRenderPassTargetDetails,
                     MTLRenderPipelineState_id,
                     MRPTDHasher>
      built_pipelines_;

 private:
  const MetalDevice *device_;

  // Compute variables
  MTLLibrary_id mtl_compute_library_;
  MTLFunction_id mtl_compute_function_;
  MTLComputePipelineState_id mtl_compute_pipeline_state_;
  MetalWorkgroupSize workgroup_size_;

  // Raster variables
  MetalRasterLibraries mtl_raster_libraries_;
  MetalRasterFunctions mtl_raster_functions_;
  MTLVertexDescriptor *vertex_descriptor_;
  MetalShaderBindingMapping binding_mapping_;
  RasterParams raster_params_;

  bool is_raster_pipeline_{false};
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

class MetalRasterResources : public RasterResources {
 public:
  explicit MetalRasterResources(MetalDevice *device) : device_(device) {
  }

  struct BufferBinding {
    MTLBuffer_id buffer{nullptr};
    size_t offset{0};
  };
  BufferBinding index_binding;
  uint32_t index_type_enum{0};

  std::unordered_map<uint32_t, BufferBinding> vertex_buffers;

  ~MetalRasterResources() override = default;

  RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) final;
  RasterResources &index_buffer(DevicePtr ptr, size_t index_width) final;

 private:
  MetalDevice *device_;
};

struct ViewportBounds {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
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
  void begin_renderpass(int x0,
                        int y0,
                        int x1,
                        int y1,
                        uint32_t num_color_attachments,
                        DeviceAllocation *color_attachments,
                        bool *color_clear,
                        std::vector<float> *clear_colors,
                        DeviceAllocation *depth_attachment,
                        bool depth_clear) override;
  void end_renderpass() override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void draw_instance(uint32_t num_verticies,
                     uint32_t num_instances,
                     uint32_t start_vertex = 0,
                     uint32_t start_instance = 0) override;
  void draw_indexed(uint32_t num_indicies,
                    uint32_t start_vertex = 0,
                    uint32_t start_index = 0) override;
  void draw_indexed_instance(uint32_t num_indicies,
                             uint32_t num_instances,
                             uint32_t start_vertex = 0,
                             uint32_t start_index = 0,
                             uint32_t start_instance = 0) override;
  void image_transition(DeviceAllocation img,
                        ImageLayout old_layout,
                        ImageLayout new_layout) final;
  void buffer_to_image(DeviceAllocation dst_img,
                       DevicePtr src_buf,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;
  void image_to_buffer(DevicePtr dst_buf,
                       DeviceAllocation src_img,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;
  void copy_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageCopyParams &params) override;
  void blit_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageCopyParams &params) override;
  void set_line_width(float width) override;

  MTLCommandBuffer_id finalize();
  // If noclear is false, ignore whatever is set in details
  // This may be used to "resume" the current renderpass
  MTLRenderPassDescriptor *create_render_pass_desc(bool depth_write,
                                                   bool noclear = false);

  bool is_renderpass_active() const;
  void set_renderpass_active();

 private:
  friend class MetalStream;

  void bind_mtl_shader_resources(MetalShaderResourceSet *resource_set,
                                 MTLRenderCommandEncoder_id rce,
                                 const MetalShaderBindingMapping *mapping);
  MTLRenderCommandEncoder_id pre_draw_setup();
  const MetalDevice *device_;
  MTLCommandBuffer_id cmdbuf_;

  // Non-null after `bind*` methods.
  const MetalPipeline *current_pipeline_;
  std::unique_ptr<MetalShaderResourceSet> current_shader_resource_set_{nullptr};
  std::unique_ptr<MetalRasterResources> current_raster_resources_{nullptr};
  MetalRenderPassTargetDetails current_renderpass_details_;
  ViewportBounds current_viewport_;
  std::vector<std::array<float, 4>> clear_colors_;
  std::vector<MTLTexture_id> render_targets_;
  MTLTexture_id depth_target_;

  // For renderpass resuming, track whether a renderpass has been started
  // Used to override LoadAction, to prevent uninteded clearing when resuming
  bool is_renderpass_active_{false};
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

class MetalSurface final : public Surface {
 public:
  MetalSurface(MetalDevice *device, const SurfaceConfig &config);
  ~MetalSurface() override;

  CAMetalLayer *mtl_layer() {
    return layer_;
  }

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;

  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::pair<uint32_t, uint32_t> get_size() override;
  int get_image_count() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;

 private:
  void destroy_swap_chain();

  SurfaceConfig config_;

  BufferFormat image_format_{BufferFormat::unknown};

  uint32_t width_{0};
  uint32_t height_{0};

  MTLTexture_id current_swap_chain_texture_;
  std::unordered_map<MTLTexture_id, DeviceAllocation> swapchain_images_;
  CAMetalDrawable_id current_drawable_;

  MetalDevice *device_{nullptr};
  CAMetalLayer *layer_;
};

constexpr BufferFormat kSwapChainImageFormat{BufferFormat::bgra8};

constexpr auto kMetalFragFunctionName = "frag_function";
constexpr auto kMetalVertFunctionName = "vert_function";

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

  std::unique_ptr<Surface> create_surface(const SurfaceConfig &config) override;

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
      std::string name = "Pipeline") override;

  RasterResources *create_raster_resources() override;

  Stream *get_compute_stream() override;
  Stream *get_graphics_stream() override;
  void wait_idle() override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  const MetalSampler &get_default_sampler() const {
    return *default_sampler_;
  }

  MTLFunction_id get_mtl_function(MTLLibrary_id mtl_lib,
                                  const std::string &func_name) const;
  MTLLibrary_id get_mtl_library(const std::string &source) const;

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
