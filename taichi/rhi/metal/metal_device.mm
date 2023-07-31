#include "taichi/rhi/metal/metal_device.h"
#include "spirv_msl.hpp"
#include "taichi/rhi/device.h"
#include "taichi/rhi/device_capability.h"
#include "taichi/rhi/impl_support.h"

namespace taichi::lang {
namespace metal {

MTLVertexFormat vertexformat2mtl(BufferFormat format) {
#define VERTEX_FORMATS
#include "taichi/rhi/metal/metal_rhi_enum_mappings.h"
#undef VERTEX_FORMATS
  auto it = map.find(format);
  RHI_ASSERT(it != map.end());
  return it->second;
}
MTLPixelFormat format2mtl(BufferFormat format) {
#define PIXEL_FORMATS
#include "taichi/rhi/metal/metal_rhi_enum_mappings.h"
#undef PIXEL_FORMATS
  auto it = map.find(format);
  RHI_ASSERT(it != map.end());
  return it->second;
}
BufferFormat mtl2format(MTLPixelFormat format) {
#define PIXEL_FORMATS
#include "taichi/rhi/metal/metal_rhi_enum_mappings.h"
#undef PIXEL_FORMATS
  auto it = std::find_if(map.begin(), map.end(),
                         [&format](auto &&p) { return p.second == format; });
  RHI_ASSERT(it != map.end());
  return it->first;
}
MTLTextureType dimension2mtl(ImageDimension dimension) {
  static const std::map<ImageDimension, MTLTextureType> map = {
      {ImageDimension::d1D, MTLTextureType1D},
      {ImageDimension::d2D, MTLTextureType2D},
      {ImageDimension::d3D, MTLTextureType3D},
  };
  auto it = map.find(dimension);
  RHI_ASSERT(it != map.end());
  return it->second;
}
MTLTextureUsage usage2mtl(ImageAllocUsage usage) {
  MTLTextureUsage out = 0;
  if (usage & ImageAllocUsage::Sampled) {
    out |= MTLTextureUsageShaderRead;
  }
  if (usage & ImageAllocUsage::Storage) {
    out |= MTLTextureUsageShaderWrite;
  }
  if (usage & ImageAllocUsage::Attachment) {
    out |= MTLTextureUsageRenderTarget;
  }
  return out;
}

MetalMemory::MetalMemory(MTLBuffer_id mtl_buffer, bool can_map)
    : mtl_buffer_(mtl_buffer), can_map_(can_map) {}
MetalMemory::~MetalMemory() {
  if (!dont_destroy_) {
    [mtl_buffer_ release];
  }
}

void MetalMemory::dont_destroy() { dont_destroy_ = true; }

MTLBuffer_id MetalMemory::mtl_buffer() const { return mtl_buffer_; }
size_t MetalMemory::size() const { return (size_t)[mtl_buffer_ length]; }
RhiResult MetalMemory::mapped_ptr(void **mapped_ptr) const {
  if (!can_map_) {
    return RhiResult::invalid_usage;
  }
  void *ptr = [mtl_buffer_ contents];
  if (ptr == nullptr) {
    return RhiResult::invalid_usage;
  } else {
    *mapped_ptr = ptr;
    return RhiResult::success;
  }
}

MetalImage::MetalImage(MTLTexture_id mtl_texture) : mtl_texture_(mtl_texture) {}
MetalImage::~MetalImage() {
  if (!dont_destroy_) {
    [mtl_texture_ release];
  }
}

void MetalImage::dont_destroy() { dont_destroy_ = true; }

MTLTexture_id MetalImage::mtl_texture() const { return mtl_texture_; }

MetalSampler::MetalSampler(MTLSamplerState_id mtl_sampler_state)
    : mtl_sampler_state_(mtl_sampler_state) {}
MetalSampler::~MetalSampler() { [mtl_sampler_state_ release]; }

MTLSamplerState_id MetalSampler::mtl_sampler_state() const {
  return mtl_sampler_state_;
}

MetalPipeline::MetalPipeline(
    const MetalDevice &device, MTLLibrary_id mtl_library,
    MTLFunction_id mtl_function,
    MTLComputePipelineState_id mtl_compute_pipeline_state,
    MetalWorkgroupSize workgroup_size)
    : device_(&device), mtl_library_(mtl_library), mtl_function_(mtl_function),
      mtl_compute_pipeline_state_(mtl_compute_pipeline_state),
      workgroup_size_(workgroup_size) {}
MetalPipeline::MetalPipeline(const MetalDevice &device,
                             MTLLibrary_id mtl_library,
                             const MetalRasterFunctions &mtl_functions,
                             MTLVertexDescriptor *vertex_descriptor,
                             const RasterParams raster_params)
    : device_(&device), mtl_library_(mtl_library),
      mtl_functions_(mtl_functions), vertex_descriptor_(vertex_descriptor),
      raster_params_(raster_params), is_raster_pipeline_(true) {}
MetalPipeline::~MetalPipeline() { destroy(); }
MetalPipeline *MetalPipeline::create_compute_pipeline(const MetalDevice &device,
                                                      const uint32_t *spv_data,
                                                      size_t spv_size,
                                                      const std::string &name) {
  RHI_ASSERT((size_t)spv_data % sizeof(uint32_t) == 0);
  RHI_ASSERT(spv_size % sizeof(uint32_t) == 0);
  spirv_cross::CompilerMSL compiler(spv_data, spv_size / sizeof(uint32_t));
  spirv_cross::CompilerMSL::Options options{};
  options.enable_decoration_binding = true;

  // Choose a proper msl version according to the device capability.
  DeviceCapabilityConfig caps = device.get_caps();
  bool feature_simd_scoped_permute_operations =
      caps.contains(DeviceCapability::spirv_has_subgroup_vote) ||
      caps.contains(DeviceCapability::spirv_has_subgroup_ballot);
  bool feature_simd_scoped_reduction_operations =
      caps.contains(DeviceCapability::spirv_has_subgroup_arithmetic);

  if (feature_simd_scoped_permute_operations ||
      feature_simd_scoped_reduction_operations) {
    // Subgroups are only supported in Metal 2.1 and up.
    options.set_msl_version(2, 1, 0);
  }
  bool feature_64_bit_integer_math =
      caps.contains(DeviceCapability::spirv_has_int64);
  if (feature_64_bit_integer_math) {
    options.set_msl_version(2, 3, 0);
  }

  compiler.set_msl_options(options);

  std::string msl = "";
  try {
    msl = compiler.compile();
  } catch (const spirv_cross::CompilerError &e) {
    std::array<char, 4096> msgbuf;
    snprintf(msgbuf.data(), msgbuf.size(), "(spirv-cross compiler) %s: %s",
             name.c_str(), e.what());
    RHI_LOG_ERROR(msgbuf.data());
    return nullptr;
  }

  MTLLibrary_id mtl_library = device.get_mtl_library(msl);

  MTLFunction_id mtl_function =
      device.get_mtl_function(mtl_library, std::string("main0"));

  MTLComputePipelineState_id mtl_compute_pipeline_state = nil;
  {
    NSError *err = nil;
    mtl_compute_pipeline_state =
        [device.mtl_device() newComputePipelineStateWithFunction:mtl_function
                                                           error:&err];

    if (mtl_compute_pipeline_state == nil) {
      if (err != nil) {
        std::array<char, 4096> msgbuf;
        snprintf(msgbuf.data(), msgbuf.size(),
                 "cannot create compute pipeline state: %s (code=%d)",
                 err.localizedDescription.UTF8String, (int)err.code);
        RHI_LOG_ERROR(msgbuf.data());
      }
      return nullptr;
    }
  }

  const spirv_cross::SPIREntryPoint &entry_point = compiler.get_entry_point(
      "main", spv::ExecutionModel::ExecutionModelGLCompute);
  MetalWorkgroupSize workgroup_size{};
  workgroup_size.x = entry_point.workgroup_size.x;
  workgroup_size.y = entry_point.workgroup_size.y;
  workgroup_size.z = entry_point.workgroup_size.z;

  return new MetalPipeline(device, mtl_library, mtl_function,
                           mtl_compute_pipeline_state,
                           std::move(workgroup_size));
}

MTLRenderPipelineState_id MetalPipeline::build_mtl_render_pipeline(
    const MetalRenderPassTargetDetails &renderpass_details) {
  // Create render pipeline
  static const std::unordered_map<TopologyType, MTLPrimitiveTopologyClass>
      topo_types = {
          {TopologyType::Triangles,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassTriangle},
          {TopologyType::Lines,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassLine},
          {TopologyType::Points,
           MTLPrimitiveTopologyClass::MTLPrimitiveTopologyClassPoint},
      };
  static const std::unordered_map<BlendOp, MTLBlendOperation> blend_ops = {
      {BlendOp::add, MTLBlendOperation::MTLBlendOperationAdd},
      {BlendOp::subtract, MTLBlendOperation::MTLBlendOperationSubtract},
      {BlendOp::reverse_subtract,
       MTLBlendOperation::MTLBlendOperationReverseSubtract},
      {BlendOp::min, MTLBlendOperation::MTLBlendOperationMin},
      {BlendOp::max, MTLBlendOperation::MTLBlendOperationMax},
  };
  static const std::unordered_map<BlendFactor, MTLBlendFactor> blend_factors = {
      {BlendFactor::zero, MTLBlendFactor::MTLBlendFactorZero},
      {BlendFactor::one, MTLBlendFactor::MTLBlendFactorOne},
      {BlendFactor::src_color, MTLBlendFactor::MTLBlendFactorSourceColor},
      {BlendFactor::one_minus_src_color,
       MTLBlendFactor::MTLBlendFactorOneMinusSourceColor},
      {BlendFactor::dst_color, MTLBlendFactor::MTLBlendFactorDestinationColor},
      {BlendFactor::one_minus_dst_color,
       MTLBlendFactor::MTLBlendFactorOneMinusDestinationColor},
      {BlendFactor::src_alpha, MTLBlendFactor::MTLBlendFactorSourceAlpha},
      {BlendFactor::one_minus_src_alpha,
       MTLBlendFactor::MTLBlendFactorOneMinusSourceAlpha},
      {BlendFactor::dst_alpha, MTLBlendFactor::MTLBlendFactorDestinationAlpha},
      {BlendFactor::one_minus_dst_alpha,
       MTLBlendFactor::MTLBlendFactorOneMinusDestinationAlpha},
  };
  MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
  rpd.vertexFunction = mtl_functions_.vertex;
  rpd.fragmentFunction = mtl_functions_.fragment;
  rpd.inputPrimitiveTopology = topo_types.at(raster_params_.prim_topology);
  rpd.vertexDescriptor = vertex_descriptor_;

  if (renderpass_details.depth_attach_format != BufferFormat::unknown) {
    rpd.depthAttachmentPixelFormat =
        format2mtl(renderpass_details.depth_attach_format);
  }

  RHI_ASSERT(raster_params_.blending.size() ==
             renderpass_details.color_attachments.size());
  for (int i = 0; i < renderpass_details.color_attachments.size(); i++) {
    MTLPixelFormat format =
        format2mtl(renderpass_details.color_attachments[i].first);
    rpd.colorAttachments[i].pixelFormat = format;

    BlendingParams blending = raster_params_.blending[i];
    rpd.colorAttachments[i].blendingEnabled = blending.enable;
    rpd.colorAttachments[i].rgbBlendOperation = blend_ops.at(blending.color.op);
    rpd.colorAttachments[i].alphaBlendOperation =
        blend_ops.at(blending.alpha.op);
    rpd.colorAttachments[i].destinationRGBBlendFactor =
        blend_factors.at(blending.color.dst_factor);
    rpd.colorAttachments[i].sourceRGBBlendFactor =
        blend_factors.at(blending.color.src_factor);
    rpd.colorAttachments[i].destinationAlphaBlendFactor =
        blend_factors.at(blending.alpha.dst_factor);
    rpd.colorAttachments[i].sourceAlphaBlendFactor =
        blend_factors.at(blending.alpha.src_factor);
  }

  MTLRenderPipelineState_id rps = nil;
  {
    NSError *err = nil;
    rps = [device_->mtl_device() newRenderPipelineStateWithDescriptor:rpd
                                                                error:&err];

    if (rps == nil) {
      if (err != nil) {
        std::array<char, 4096> msgbuf;
        snprintf(msgbuf.data(), msgbuf.size(),
                 "cannot create render pipeline state: %s (code=%d)",
                 err.localizedDescription.UTF8String, (int)err.code);
        RHI_LOG_ERROR(msgbuf.data());
      }
      return nullptr;
    }
  }

  return rps;
}

void MetalPipeline::destroy() {
  if (!is_destroyed_) {
    [mtl_compute_pipeline_state_ release];
    [mtl_function_ release];
    [mtl_library_ release];
    is_destroyed_ = true;
  }
}

MetalShaderResourceSet::MetalShaderResourceSet(const MetalDevice &device)
    : device_(&device) {}
MetalShaderResourceSet::~MetalShaderResourceSet() {}

ShaderResourceSet &MetalShaderResourceSet::buffer(uint32_t binding,
                                                  DevicePtr ptr, size_t size) {
  RHI_ASSERT(ptr.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = ptr.offset;
  rsc.buffer.size = size;
  resources_.push_back(std::move(rsc));

  return *this;
}
ShaderResourceSet &MetalShaderResourceSet::buffer(uint32_t binding,
                                                  DeviceAllocation alloc) {
  RHI_ASSERT(alloc.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = 0;
  rsc.buffer.size = memory.size();
  resources_.push_back(std::move(rsc));

  return *this;
}

ShaderResourceSet &MetalShaderResourceSet::rw_buffer(uint32_t binding,
                                                     DevicePtr ptr,
                                                     size_t size) {
  RHI_ASSERT(ptr.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = ptr.offset;
  rsc.buffer.size = size;
  resources_.push_back(std::move(rsc));

  return *this;
}
ShaderResourceSet &MetalShaderResourceSet::rw_buffer(uint32_t binding,
                                                     DeviceAllocation alloc) {
  RHI_ASSERT(alloc.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = 0;
  rsc.buffer.size = memory.size();
  resources_.push_back(std::move(rsc));

  return *this;
}

ShaderResourceSet &
MetalShaderResourceSet::image(uint32_t binding, DeviceAllocation alloc,
                              ImageSamplerConfig sampler_config) {
  RHI_ASSERT(alloc.device == (Device *)device_);
  const MetalImage &image = device_->get_image(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::texture;
  rsc.binding = binding;
  rsc.texture.texture = image.mtl_texture();
  rsc.texture.is_sampled = true;
  resources_.push_back(std::move(rsc));

  return *this;
}

ShaderResourceSet &MetalShaderResourceSet::rw_image(uint32_t binding,
                                                    DeviceAllocation alloc,
                                                    int lod) {
  RHI_ASSERT(alloc.device == (Device *)device_);
  const MetalImage &image = device_->get_image(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::texture;
  rsc.binding = binding;
  rsc.texture.texture = image.mtl_texture();
  rsc.texture.is_sampled = false;
  resources_.push_back(std::move(rsc));

  return *this;
}

RasterResources &MetalRasterResources::vertex_buffer(DevicePtr ptr,
                                                     uint32_t binding) {
  MTLBuffer_id buffer = (ptr != kDeviceNullPtr)
                            ? device_->get_memory(ptr.alloc_id).mtl_buffer()
                            : nullptr;
  if (buffer == nullptr) {
    vertex_buffers.erase(binding);
  } else {
    vertex_buffers[binding] = {buffer, ptr.offset};
  }
  return *this;
}

RasterResources &MetalRasterResources::index_buffer(DevicePtr ptr,
                                                    size_t index_width) {
  MTLBuffer_id buffer = (ptr != kDeviceNullPtr)
                            ? device_->get_memory(ptr.alloc_id).mtl_buffer()
                            : nullptr;
  if (buffer == nullptr) {
    index_binding = BufferBinding();
  } else {
    index_binding = {buffer, ptr.offset};
  }
  return *this;
}
MetalCommandList::MetalCommandList(const MetalDevice &device,
                                   MTLCommandQueue_id cmd_queue)
    : device_(&device) {
  @autoreleasepool {
    cmdbuf_ = [cmd_queue commandBuffer];
    [cmdbuf_ retain];
  }
}

MetalCommandList::~MetalCommandList() { [cmdbuf_ release]; }

void MetalCommandList::bind_pipeline(Pipeline *p) noexcept {
  RHI_ASSERT(p != nullptr);
  current_pipeline_ = (MetalPipeline *)p;

  auto pipeline = static_cast<MetalPipeline *>(p);

  if (pipeline->is_graphics()) {
    // Check if PSO is already built for current render pass parameters
    if (pipeline->built_pipelines_.count(current_renderpass_details_) == 0) {
      // Not built, need to build
      MTLRenderPipelineState_id rps =
          pipeline->build_mtl_render_pipeline(current_renderpass_details_);
      pipeline->built_pipelines_[current_renderpass_details_] = rps;
    }
  }
}
RhiResult MetalCommandList::bind_shader_resources(ShaderResourceSet *res,
                                                  int set_index) noexcept {
  RHI_ASSERT(res != nullptr);
  RHI_ASSERT(set_index == 0);
  current_shader_resource_set_ = (MetalShaderResourceSet *)res;
  return RhiResult::success;
}

RhiResult
MetalCommandList::bind_raster_resources(RasterResources *res) noexcept {
  return RhiResult::not_supported;
}

void MetalCommandList::buffer_barrier(DeviceAllocation alloc) noexcept {}
void MetalCommandList::buffer_barrier(DevicePtr ptr, size_t size) noexcept {}
void MetalCommandList::memory_barrier() noexcept {
  // NOTE: (penguinliong) Resources created from `MTLDevice` (which is the only
  // available way to allocate resource here) are `MTLHazardTrackingModeTracked`
  // by default. So we don't have to barrier explicitly.
}

void MetalCommandList::buffer_copy(DevicePtr dst, DevicePtr src,
                                   size_t size) noexcept {
  const MetalMemory &src_memory = device_->get_memory(src.alloc_id);
  const MetalMemory &dst_memory = device_->get_memory(dst.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size_t src_size = src_memory.size();
    size_t dst_size = dst_memory.size();
    RHI_ASSERT(src_size == dst_size);
    size = src_size;
  }

  MTLBuffer_id src_mtl_buffer = src_memory.mtl_buffer();
  MTLBuffer_id dst_mtl_buffer = dst_memory.mtl_buffer();

  @autoreleasepool {
    MTLBlitCommandEncoder_id encoder = [cmdbuf_ blitCommandEncoder];
    [encoder copyFromBuffer:src_mtl_buffer
               sourceOffset:NSUInteger(src.offset)
                   toBuffer:dst_mtl_buffer
          destinationOffset:NSUInteger(dst.offset)
                       size:size];
    [encoder endEncoding];
  }
}

void MetalCommandList::buffer_fill(DevicePtr ptr, size_t size,
                                   uint32_t data) noexcept {
  RHI_ASSERT(data == 0);

  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size = memory.size();
  }

  MTLBuffer_id mtl_buffer = memory.mtl_buffer();

  @autoreleasepool {
    MTLBlitCommandEncoder_id encoder = [cmdbuf_ blitCommandEncoder];
    [encoder fillBuffer:mtl_buffer
                  range:NSMakeRange((NSUInteger)ptr.offset, (NSUInteger)size)
                  value:0];
    [encoder endEncoding];
  }
}

RhiResult MetalCommandList::dispatch(uint32_t x, uint32_t y,
                                     uint32_t z) noexcept {
  RHI_ASSERT(current_pipeline_);
  RHI_ASSERT(current_shader_resource_set_);

  MTLComputePipelineState_id mtl_compute_pipeline_state =
      current_pipeline_->mtl_compute_pipeline_state();

  NSUInteger local_x = current_pipeline_->workgroup_size().x;
  NSUInteger local_y = current_pipeline_->workgroup_size().y;
  NSUInteger local_z = current_pipeline_->workgroup_size().z;

  std::vector<MetalShaderResource> shader_resources =
      current_shader_resource_set_->resources();

  @autoreleasepool {
    MTLComputeCommandEncoder_id encoder = [cmdbuf_ computeCommandEncoder];

    for (const MetalShaderResource &resource : shader_resources) {
      switch (resource.ty) {
      case MetalShaderResourceType::buffer: {
        [encoder setBuffer:resource.buffer.buffer
                    offset:resource.buffer.offset
                   atIndex:resource.binding];
        break;
      }
      case MetalShaderResourceType::texture: {
        [encoder setTexture:resource.texture.texture atIndex:resource.binding];
        if (resource.texture.is_sampled) {
          [encoder
              setSamplerState:device_->get_default_sampler().mtl_sampler_state()
                      atIndex:resource.binding];
        }
        break;
      }
      default:
        RHI_ASSERT(false);
      }
    }

    [encoder setComputePipelineState:mtl_compute_pipeline_state];
    [encoder dispatchThreadgroups:MTLSizeMake(x, y, z)
            threadsPerThreadgroup:MTLSizeMake(local_x, local_y, local_z)];
    [encoder endEncoding];
  };

  return RhiResult::success;
}

void MetalCommandList::begin_renderpass(int x0, int y0, int x1, int y1,
                                        uint32_t num_color_attachments,
                                        DeviceAllocation *color_attachments,
                                        bool *color_clear,
                                        std::vector<float> *clear_colors,
                                        DeviceAllocation *depth_attachment,
                                        bool depth_clear) {
  // TODO: Create MTLViewPort object for the x0, y0, x1, y1 parameters

  current_renderpass_details_.color_attachments.clear();
  current_renderpass_details_.clear_depth = depth_clear;

  if (depth_attachment) {
    MetalImage depth_attach = device_->get_image(depth_attachment->alloc_id);
    BufferFormat format = mtl2format(depth_attach.mtl_texture().pixelFormat);
    current_renderpass_details_.depth_attach_format = format;
  }

  for (int i = 0; i < num_color_attachments; i++) {
    MetalImage col_attach = device_->get_image(color_attachments[i].alloc_id);
    BufferFormat format = mtl2format(col_attach.mtl_texture().pixelFormat);
    bool clear = color_clear[i];
    current_renderpass_details_.color_attachments.emplace_back(format, clear);
  }
  clear_colors_ = clear_colors;
}

void MetalCommandList::image_transition(DeviceAllocation img,
                                        ImageLayout old_layout,
                                        ImageLayout new_layout) {}

MTLCommandBuffer_id MetalCommandList::finalize() { return cmdbuf_; }

MetalStream::MetalStream(const MetalDevice &device,
                         MTLCommandQueue_id mtl_command_queue)
    : device_(&device), mtl_command_queue_(mtl_command_queue) {}
MetalStream::~MetalStream() { destroy(); }

MetalStream *MetalStream::create(const MetalDevice &device) {
  MTLCommandQueue_id compute_queue = [device.mtl_device() newCommandQueue];
  return new MetalStream(device, compute_queue);
}
void MetalStream::destroy() {
  if (!is_destroyed_) {
    command_sync();
    [mtl_command_queue_ release];
    is_destroyed_ = true;
  }
}

RhiResult MetalStream::new_command_list(CommandList **out_cmdlist) noexcept {
  *out_cmdlist = new MetalCommandList(*device_, mtl_command_queue_);
  return RhiResult::success;
}

StreamSemaphore
MetalStream::submit(CommandList *cmdlist,
                    const std::vector<StreamSemaphore> &wait_semaphores) {
  MetalCommandList *cmdlist2 = (MetalCommandList *)cmdlist;

  MTLCommandBuffer_id cmdbuf = [cmdlist2->finalize() retain];
  [cmdbuf commit];
  pending_cmdbufs_.push_back(cmdbuf);

  return {};
}
StreamSemaphore MetalStream::submit_synced(
    CommandList *cmdlist, const std::vector<StreamSemaphore> &wait_semaphores) {
  auto sema = submit(cmdlist, wait_semaphores);
  command_sync();
  return sema;
}
void MetalStream::command_sync() {
  for (const auto &cmdbuf : pending_cmdbufs_) {
    [cmdbuf waitUntilCompleted];
    [cmdbuf release];
  }
  pending_cmdbufs_.clear();
}

DeviceCapabilityConfig collect_metal_device_caps(MTLDevice_id mtl_device) {
  // https://developer.apple.com/documentation/metal/mtlgpufamily/mtlgpufamilyapple8?language=objc
  // We do this so that it compiles under lower version of macOS
  [[maybe_unused]] constexpr auto kMTLGPUFamilyApple8 = MTLGPUFamily(1008);
  constexpr auto kMTLGPUFamilyApple7 = MTLGPUFamily(1007);
  constexpr auto kMTLGPUFamilyApple6 = MTLGPUFamily(1006);
  constexpr auto kMTLGPUFamilyApple5 = MTLGPUFamily(1005);
  constexpr auto kMTLGPUFamilyApple4 = MTLGPUFamily(1004);
  constexpr auto kMTLGPUFamilyApple3 = MTLGPUFamily(1003);

  bool family_mac2 = [mtl_device supportsFamily:MTLGPUFamilyMac2];
  bool family_apple7 = [mtl_device supportsFamily:kMTLGPUFamilyApple7];
  bool family_apple6 =
      [mtl_device supportsFamily:kMTLGPUFamilyApple6] | family_apple7;
  bool family_apple5 =
      [mtl_device supportsFamily:kMTLGPUFamilyApple5] | family_apple6;
  bool family_apple4 =
      [mtl_device supportsFamily:kMTLGPUFamilyApple4] | family_apple5;
  bool family_apple3 =
      [mtl_device supportsFamily:kMTLGPUFamilyApple3] | family_apple4;

  bool feature_64_bit_integer_math = family_apple3;
  bool feature_floating_point_atomics = family_apple7 | family_mac2;
  bool feature_quad_scoped_permute_operations = family_apple4 | family_mac2;
  bool feature_simd_scoped_permute_operations = family_apple6 | family_mac2;
  bool feature_simd_scoped_reduction_operations = family_apple7 | family_mac2;

  DeviceCapabilityConfig caps{};
  caps.set(DeviceCapability::spirv_version, 0x10300);
  caps.set(DeviceCapability::spirv_has_int8, 1);
  caps.set(DeviceCapability::spirv_has_int16, 1);
  caps.set(DeviceCapability::spirv_has_float16, 1);
  caps.set(DeviceCapability::spirv_has_subgroup_basic, 1);

  if (feature_64_bit_integer_math) {
    caps.set(DeviceCapability::spirv_has_int64, 1);
  }
  if (feature_floating_point_atomics) {
    // FIXME: (penguinliong) For some reason floating point atomics doesn't
    // work and breaks the FEM99/FEM128 examples. Should consider add them back
    // figured out why.
    // caps.set(DeviceCapability::spirv_has_atomic_float, 1);
    // caps.set(DeviceCapability::spirv_has_atomic_float_add, 1);
  }
  if (feature_simd_scoped_permute_operations ||
      feature_quad_scoped_permute_operations) {
    caps.set(DeviceCapability::spirv_has_subgroup_vote, 1);
    caps.set(DeviceCapability::spirv_has_subgroup_ballot, 1);
  }
  if (feature_simd_scoped_reduction_operations) {
    caps.set(DeviceCapability::spirv_has_subgroup_arithmetic, 1);
  }
  return caps;
}

std::unique_ptr<MetalSampler> create_sampler(id<MTLDevice> mtl_device) {
  MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
  desc.magFilter = MTLSamplerMinMagFilterLinear;
  desc.minFilter = MTLSamplerMinMagFilterLinear;
  desc.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
  desc.tAddressMode = MTLSamplerAddressModeMirrorRepeat;
  desc.rAddressMode = MTLSamplerAddressModeMirrorRepeat;
  desc.compareFunction = MTLCompareFunctionAlways;
  desc.mipFilter = MTLSamplerMipFilterLinear;

  id<MTLSamplerState> mtl_sampler_state =
      [mtl_device newSamplerStateWithDescriptor:desc];

  [desc release];

  return std::make_unique<MetalSampler>(mtl_sampler_state);
}

MetalSurface::MetalSurface(MetalDevice *device, const SurfaceConfig &config)
    : config_(config), device_(device) {

  width_ = config.width;
  height_ = config.height;

  image_format_ = kSwapChainImageFormat;

  layer_ = [CAMetalLayer layer];
  layer_.device = device->mtl_device();
  layer_.pixelFormat = format2mtl(image_format_);
  layer_.drawableSize = CGSizeMake(width_, height_);
  layer_.allowsNextDrawableTimeout = NO;
#if TARGET_OS_OSX
  // Older versions may not have this property so check if it exists first.
  layer_.displaySyncEnabled = config.vsync;
#endif
}

MetalSurface::~MetalSurface() {
  destroy_swap_chain();
  [layer_ release];
}

void MetalSurface::destroy_swap_chain() {
  for (auto &alloc : swapchain_images_) {
    device_->destroy_image(alloc.second);
  }
  swapchain_images_.clear();
}

StreamSemaphore MetalSurface::acquire_next_image() {
  current_drawable_ = [layer_ nextDrawable];
  current_swap_chain_texture_ = current_drawable_.texture;

  if (swapchain_images_.count(current_swap_chain_texture_) == 0) {
    swapchain_images_[current_swap_chain_texture_] =
        device_->import_mtl_texture(current_drawable_.texture);
    RHI_ASSERT(swapchain_images_.size() <=
               50); // In case something goes wrong on Metal side, prevent this
                    // map of images from growing each frame unbounded.
  }
  return nullptr;
}

DeviceAllocation MetalSurface::get_target_image() {
  return swapchain_images_.at(current_swap_chain_texture_);
}

void MetalSurface::present_image(
    const std::vector<StreamSemaphore> &wait_semaphores) {

  [current_drawable_ present];

  device_->wait_idle();
}

std::pair<uint32_t, uint32_t> MetalSurface::get_size() {
  return std::make_pair(width_, height_);
}

int MetalSurface::get_image_count() { return (int)layer_.maximumDrawableCount; }

BufferFormat MetalSurface::image_format() { return image_format_; }

void MetalSurface::resize(uint32_t width, uint32_t height) {
  destroy_swap_chain();
  width_ = width;
  height_ = height;
  layer_.drawableSize = CGSizeMake(width_, height_);
}

MetalDevice::MetalDevice(MTLDevice_id mtl_device) : mtl_device_(mtl_device) {
  compute_stream_ = std::unique_ptr<MetalStream>(MetalStream::create(*this));

  default_sampler_ = create_sampler(mtl_device);

  DeviceCapabilityConfig caps = collect_metal_device_caps(mtl_device);
  set_caps(std::move(caps));
}
MetalDevice::~MetalDevice() { destroy(); }

MetalDevice *MetalDevice::create() {
  MTLDevice_id mtl_device = MTLCreateSystemDefaultDevice();

  return new MetalDevice(mtl_device);
}
void MetalDevice::destroy() {
  if (!is_destroyed_) {
    compute_stream_.reset();
    memory_allocs_.clear();
    [mtl_device_ release];
    is_destroyed_ = true;
  }
}

std::unique_ptr<Surface>
MetalDevice::create_surface(const SurfaceConfig &config) {
  return std::make_unique<MetalSurface>(this, config);
}

RhiResult MetalDevice::allocate_memory(const AllocParams &params,
                                       DeviceAllocation *out_devalloc) {
  if (params.export_sharing) {
    RHI_LOG_ERROR("export sharing is not available in metal");
  }

  bool can_map = params.host_read || params.host_write;

  MTLStorageMode storage_mode;
  if (can_map) {
    storage_mode = MTLStorageModeShared;
  } else {
    storage_mode = MTLStorageModePrivate;
  }
  MTLCPUCacheMode cpu_cache_mode = MTLCPUCacheModeDefaultCache;
  MTLResourceOptions resource_options =
      (storage_mode << MTLResourceStorageModeShift) |
      (cpu_cache_mode << MTLResourceCPUCacheModeShift);

  MTLBuffer_id buffer = [mtl_device_ newBufferWithLength:params.size
                                                 options:resource_options];

  MetalMemory &alloc = memory_allocs_.acquire(buffer, can_map);

  *out_devalloc = DeviceAllocation{};
  out_devalloc->device = this;
  out_devalloc->alloc_id = reinterpret_cast<uint64_t>(&alloc);

  return RhiResult::success;
}
DeviceAllocation MetalDevice::import_mtl_buffer(MTLBuffer_id buffer) {
  bool can_map = [buffer contents] != nullptr;
  MetalMemory &alloc = memory_allocs_.acquire(buffer, can_map);
  alloc.dont_destroy();

  DeviceAllocation out{};
  out.device = this;
  out.alloc_id = reinterpret_cast<uint64_t>(&alloc);
  return out;
}
void MetalDevice::dealloc_memory(DeviceAllocation handle) {
  RHI_ASSERT(handle.device == this);
  memory_allocs_.release(&get_memory(handle.alloc_id));
}

DeviceAllocation MetalDevice::create_image(const ImageParams &params) {
  if (params.export_sharing) {
    RHI_LOG_ERROR("export sharing is not available in metal");
  }

  MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
  desc.width = params.x;
  desc.height = params.y;
  desc.depth = params.z;
  desc.arrayLength = 1;
  desc.pixelFormat = format2mtl(params.format);
  desc.textureType = dimension2mtl(params.dimension);
  desc.usage = usage2mtl(params.usage);

  MTLTexture_id mtl_texture = [mtl_device_ newTextureWithDescriptor:desc];

  [desc release];

  MetalImage &alloc = image_allocs_.acquire(mtl_texture);

  DeviceAllocation out{};
  out.device = this;
  out.alloc_id = reinterpret_cast<uint64_t>(&alloc);
  return out;
}
DeviceAllocation MetalDevice::import_mtl_texture(MTLTexture_id texture) {
  MetalImage &alloc = image_allocs_.acquire(texture);
  alloc.dont_destroy();

  DeviceAllocation out{};
  out.device = this;
  out.alloc_id = reinterpret_cast<uint64_t>(&alloc);
  return out;
}
void MetalDevice::destroy_image(DeviceAllocation handle) {
  RHI_ASSERT(handle.device == this);
  image_allocs_.release(&get_image(handle.alloc_id));
}

const MetalMemory &MetalDevice::get_memory(DeviceAllocationId alloc_id) const {
  return *reinterpret_cast<MetalMemory *>(alloc_id);
}

MetalMemory &MetalDevice::get_memory(DeviceAllocationId alloc_id) {
  return *reinterpret_cast<MetalMemory *>(alloc_id);
}

const MetalImage &MetalDevice::get_image(DeviceAllocationId alloc_id) const {
  return *reinterpret_cast<MetalImage *>(alloc_id);
}

MetalImage &MetalDevice::get_image(DeviceAllocationId alloc_id) {
  return *reinterpret_cast<MetalImage *>(alloc_id);
}

RhiResult MetalDevice::map_range(DevicePtr ptr, uint64_t size,
                                 void **mapped_ptr) {
  const MetalMemory &memory = get_memory(ptr.alloc_id);

  size_t offset = (size_t)ptr.offset;
  RHI_ASSERT(offset + size <= memory.size());

  RhiResult result = map(ptr, mapped_ptr);
  *(const uint8_t **)mapped_ptr += offset;
  return result;
}
RhiResult MetalDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  const MetalMemory &memory = get_memory(alloc.alloc_id);
  return memory.mapped_ptr(mapped_ptr);
}
void MetalDevice::unmap(DevicePtr ptr) {}
void MetalDevice::unmap(DeviceAllocation ptr) {}

RhiResult MetalDevice::create_pipeline(Pipeline **out_pipeline,
                                       const PipelineSourceDesc &src,
                                       std::string name,
                                       PipelineCache *cache) noexcept {
  RHI_ASSERT(src.type == PipelineSourceType::spirv_binary);
  try {
    *out_pipeline = MetalPipeline::create_compute_pipeline(
        *this, (const uint32_t *)src.data, src.size, name);
  } catch (const std::exception &e) {
    return RhiResult::error;
  }
  return RhiResult::success;
}

std::unique_ptr<Pipeline> MetalDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs, std::string name) {

  // (geometry shaders aren't supported in Vulkan backend either)
  RHI_ASSERT(src.size() == 2);
  bool has_fragment = false;
  bool has_vertex = false;
  for (auto &pipe_source_desc : src) {
    RHI_ASSERT(pipe_source_desc.type == PipelineSourceType::spirv_binary);
    if (pipe_source_desc.stage == PipelineStageType::fragment)
      has_fragment = true;
    if (pipe_source_desc.stage == PipelineStageType::vertex)
      has_vertex = true;
  }
  RHI_ASSERT(has_fragment && has_vertex);

  spirv_cross::CompilerMSL::Options options{};
  options.enable_decoration_binding = true;

  // Compile spirv binaries to MSL source
  std::string msl_source = "";
  for (int i = 0; i < 2; i++) {
    const uint32_t *spv_data = (const uint32_t *)src[i].data;

    RHI_ASSERT((size_t)spv_data % sizeof(uint32_t) == 0);
    RHI_ASSERT(src[i].size % sizeof(uint32_t) == 0);

    spirv_cross::CompilerMSL compiler(spv_data, src[i].size / sizeof(uint32_t));
    compiler.set_msl_options(options);

    std::string msl = "";
    try {
      msl = compiler.compile();
    } catch (const spirv_cross::CompilerError &e) {
      std::array<char, 4096> msgbuf;
      snprintf(msgbuf.data(), msgbuf.size(), "(spirv-cross compiler) %s: %s",
               name.c_str(), e.what());
      RHI_LOG_ERROR(msgbuf.data());
      return nullptr;
    }

    const std::string new_entry_point_name =
        src[i].stage == PipelineStageType::fragment
            ? std::string(kMetalFragFunctionName)
            : std::string(kMetalVertFunctionName);

    // Fragment and vertex function names must be different.
    // If spirv-cross has a method to let you set the emitted entry point's
    // name, that would be nice, but I could not find anything in the docs.
    // So for now just using std's regex_replace().
    std::regex entry_point_regex("main0");
    msl_source +=
        std::regex_replace(msl, entry_point_regex, new_entry_point_name);
  }

  // Compile MSL source
  MTLLibrary_id mtl_library = get_mtl_library(msl_source);

  // Get the MTLFunctions
  int frag_func_index = 0;
  int vert_func_index = 0;
  std::vector<MTLFunction_id> mtl_functions;
  for (int i = 0; i < 2; i++) {
    std::string new_entry_point_name;
    if (src[i].stage == PipelineStageType::fragment) {
      frag_func_index = i;
      new_entry_point_name = std::string(kMetalFragFunctionName);
    } else {
      vert_func_index = i;
      new_entry_point_name = std::string(kMetalVertFunctionName);
    }

    MTLFunction_id mtl_function =
        get_mtl_function(mtl_library, new_entry_point_name);
    mtl_functions.push_back(mtl_function);
  }

  // Set vertex description
  MTLVertexDescriptor *vd = [MTLVertexDescriptor new];

  for (auto &vert_attr : vertex_attrs) {
    int location = vert_attr.location;
    vd.attributes[location].format = vertexformat2mtl(vert_attr.format);
    vd.attributes[location].offset = vert_attr.offset;
    vd.attributes[location].bufferIndex = vert_attr.binding;
  }
  for (auto &vert_input : vertex_inputs) {
    int buffer_index = vert_input.binding;
    vd.layouts[buffer_index].stride = vert_input.stride;
    vd.layouts[buffer_index].stepFunction =
        vert_input.instance ? MTLVertexStepFunctionPerInstance
                            : MTLVertexStepFunctionPerVertex;
    vd.layouts[buffer_index].stepRate = 1;
  }

  [vd retain];

  // Create the pipeline object
  return std::make_unique<MetalPipeline>(
      *this, mtl_library,
      MetalRasterFunctions{mtl_functions[vert_func_index],
                           mtl_functions[frag_func_index]},
      vd, raster_params);
}

MTLFunction_id
MetalDevice::get_mtl_function(MTLLibrary_id mtl_lib,
                              const std::string &func_name) const {

  MTLFunction_id mtl_function = nil;
  NSString *entry_name_ns =
      [[NSString alloc] initWithUTF8String:func_name.c_str()];
  mtl_function = [mtl_lib newFunctionWithName:entry_name_ns];
  [entry_name_ns release];
  if (mtl_function == nil) {
    RHI_LOG_ERROR("cannot extract entry point function from shader library");
  }
  return mtl_function;
}

MTLLibrary_id MetalDevice::get_mtl_library(const std::string &source) const {
  MTLLibrary_id mtl_library = nil;
  NSError *err = nil;
  NSString *msl_ns = [[NSString alloc] initWithUTF8String:source.c_str()];
  mtl_library = [mtl_device_ newLibraryWithSource:msl_ns
                                          options:nil
                                            error:&err];
  [msl_ns release];

  if (mtl_library == nil) {
    if (err != nil) {
      std::array<char, 4096> msgbuf;
      snprintf(msgbuf.data(), msgbuf.size(),
               "cannot compile metal library from source: %s (code=%d)",
               err.localizedDescription.UTF8String, (int)err.code);
      RHI_LOG_ERROR(msgbuf.data());
    }
    return nil;
  }
  return mtl_library;
}

ShaderResourceSet *MetalDevice::create_resource_set() {
  return new MetalShaderResourceSet(*this);
}

RasterResources *MetalDevice::create_raster_resources() {
  return new MetalRasterResources(this);
}

Stream *MetalDevice::get_compute_stream() { return compute_stream_.get(); }
Stream *MetalDevice::get_graphics_stream() {
  // FIXME: (penguinliong) Support true multistream in the future. We need a
  // working semaphore.
  return compute_stream_.get();
}
void MetalDevice::wait_idle() { compute_stream_->command_sync(); }

void MetalDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  Stream *stream = get_compute_stream();
  auto [cmd, res] = stream->new_command_list_unique();
  RHI_ASSERT(res == RhiResult::success);
  cmd->buffer_copy(dst, src, size);
  stream->submit_synced(cmd.get());
}

} // namespace metal
} // namespace taichi::lang
