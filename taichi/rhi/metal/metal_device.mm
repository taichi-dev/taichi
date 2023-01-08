#include "taichi/rhi/metal/metal_device.h"
#include "spirv_msl.hpp"
#include "taichi/rhi/device_capability.h"

namespace taichi::lang {
namespace metal {

MetalMemory::MetalMemory(MTLBuffer_id mtl_buffer) : mtl_buffer_(mtl_buffer) {}
MetalMemory::~MetalMemory() { [mtl_buffer_ release]; }

MTLBuffer_id MetalMemory::mtl_buffer() const { return mtl_buffer_; }
size_t MetalMemory::size() const { return (size_t)[mtl_buffer_ length]; }
RhiResult MetalMemory::mapped_ptr(void **mapped_ptr) const {
  void *ptr = [mtl_buffer_ contents];
  if (ptr == nullptr) {
    return RhiResult::invalid_usage;
  } else {
    *mapped_ptr = ptr;
    return RhiResult::success;
  }
}

MetalPipeline::MetalPipeline(
    const MetalDevice &device, MTLLibrary_id mtl_library,
    MTLFunction_id mtl_function,
    MTLComputePipelineState_id mtl_compute_pipeline_state,
    MetalWorkgroupSize workgroup_size)
    : device_(&device), mtl_library_(mtl_library), mtl_function_(mtl_function),
      mtl_compute_pipeline_state_(mtl_compute_pipeline_state),
      workgroup_size_(workgroup_size) {}
MetalPipeline::~MetalPipeline() { destroy(); }
MetalPipeline *MetalPipeline::create(const MetalDevice &device,
                                     const uint32_t *spv_data,
                                     size_t spv_size) {
  TI_ASSERT((size_t)spv_data % sizeof(uint32_t) == 0);
  TI_ASSERT(spv_size % sizeof(uint32_t) == 0);
  spirv_cross::CompilerMSL compiler(spv_data, spv_size / sizeof(uint32_t));
  spirv_cross::CompilerMSL::Options options{};
  options.enable_decoration_binding = true;
  compiler.set_msl_options(options);
  std::string msl = compiler.compile();

  MTLLibrary_id mtl_library = nil;
  {
    NSError *err = nil;
    NSString *msl_ns = [[NSString alloc] initWithUTF8String:msl.c_str()];
    mtl_library = [device.mtl_device() newLibraryWithSource:msl_ns
                                                    options:nil
                                                      error:&err];

    if (mtl_library == nil) {
      TI_WARN_IF(err != nil,
                 "cannot compile metal library from source: {} (code={})",
                 err.localizedDescription.UTF8String, (uint32_t)err.code);
      return nullptr;
    }
  }

  MTLFunction_id mtl_function = nil;
  {
    NSString *entry_name_ns = [[NSString alloc] initWithUTF8String:"main0"];
    mtl_function = [mtl_library newFunctionWithName:entry_name_ns];
    if (mtl_library == nil) {
      // FIXME: (penguinliong) Specify the actual entry name after we compile
      // directly to MSL in codegen.
      TI_WARN("cannot extract entry point function '{}' from shader library",
              "main");
    }
  }

  MTLComputePipelineState_id mtl_compute_pipeline_state = nil;
  {
    NSError *err = nil;
    mtl_compute_pipeline_state =
        [device.mtl_device() newComputePipelineStateWithFunction:mtl_function
                                                           error:&err];

    if (mtl_compute_pipeline_state == nil) {
      TI_WARN_IF(err != nil,
                 "cannot create compute pipeline state: {} (code={})",
                 err.localizedDescription.UTF8String, (uint32_t)err.code);
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
  TI_ASSERT(ptr.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = ptr.offset;
  rsc.buffer.size = size;
  resources_.emplace_back(std::move(rsc));

  return *this;
}
ShaderResourceSet &MetalShaderResourceSet::buffer(uint32_t binding,
                                                  DeviceAllocation alloc) {
  TI_ASSERT(alloc.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = 0;
  rsc.buffer.size = memory.size();
  resources_.emplace_back(std::move(rsc));

  return *this;
}

ShaderResourceSet &MetalShaderResourceSet::rw_buffer(uint32_t binding,
                                                     DevicePtr ptr,
                                                     size_t size) {
  TI_ASSERT(ptr.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = ptr.offset;
  rsc.buffer.size = size;
  resources_.emplace_back(std::move(rsc));

  return *this;
}
ShaderResourceSet &MetalShaderResourceSet::rw_buffer(uint32_t binding,
                                                     DeviceAllocation alloc) {
  TI_ASSERT(alloc.device == (Device *)device_);
  const MetalMemory &memory = device_->get_memory(alloc.alloc_id);

  MetalShaderResource rsc{};
  rsc.ty = MetalShaderResourceType::buffer;
  rsc.binding = binding;
  rsc.buffer.buffer = memory.mtl_buffer();
  rsc.buffer.offset = 0;
  rsc.buffer.size = memory.size();
  resources_.emplace_back(std::move(rsc));

  return *this;
}

MetalCommandList::MetalCommandList(const MetalDevice &device)
    : device_(&device) {}
MetalCommandList::~MetalCommandList() {}

void MetalCommandList::bind_pipeline(Pipeline *p) noexcept {
  TI_ASSERT(p != nullptr);
  current_pipeline_ = (MetalPipeline *)p;
}
RhiResult MetalCommandList::bind_shader_resources(ShaderResourceSet *res,
                                                  int set_index) noexcept {
  TI_ASSERT(res != nullptr);
  TI_ASSERT(set_index == 0);
  current_shader_resource_set_ = (MetalShaderResourceSet *)res;
  return RhiResult::success;
}

RhiResult
MetalCommandList::bind_raster_resources(RasterResources *res) noexcept {
  return RhiResult::not_supported;
}

void MetalCommandList::memory_barrier() noexcept {
  // Note that resources created from `MTLDevice` (which is the only available
  // way to allocate resource here) are `MTLHazardTrackingModeTracked` by
  // default. So we don't have to barrier explicitly.
}

void MetalCommandList::buffer_copy(DevicePtr dst, DevicePtr src,
                                   size_t size) noexcept {
  const MetalMemory &src_memory = device_->get_memory(src.alloc_id);
  const MetalMemory &dst_memory = device_->get_memory(dst.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size_t src_size = src_memory.size();
    size_t dst_size = dst_memory.size();
    TI_ASSERT(src_size == dst_size);
    size = src_size;
  }

  MTLBuffer_id src_mtl_buffer = src_memory.mtl_buffer();
  MTLBuffer_id dst_mtl_buffer = dst_memory.mtl_buffer();

  auto encode_f = [=](MTLCommandBuffer_id mtl_command_buffer) {
    MTLBlitCommandEncoder_id encoder = [mtl_command_buffer blitCommandEncoder];
    [encoder copyFromBuffer:src_mtl_buffer
               sourceOffset:(NSUInteger)src.offset
                   toBuffer:dst_mtl_buffer
          destinationOffset:(NSUInteger)dst.offset
                       size:size];
    [encoder endEncoding];
  };
  pending_commands_.emplace_back(encode_f);
}
void MetalCommandList::buffer_fill(DevicePtr ptr, size_t size,
                                   uint32_t data) noexcept {
  TI_ASSERT(data == 0);

  const MetalMemory &memory = device_->get_memory(ptr.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size = memory.size();
  }

  MTLBuffer_id mtl_buffer = memory.mtl_buffer();

  auto encode_f = [=](MTLCommandBuffer_id mtl_command_buffer) {
    MTLBlitCommandEncoder_id encoder = [mtl_command_buffer blitCommandEncoder];
    [encoder fillBuffer:mtl_buffer
                  range:NSMakeRange((NSUInteger)ptr.offset, (NSUInteger)size)
                  value:0];
    [encoder endEncoding];
  };
  pending_commands_.emplace_back(encode_f);
}

RhiResult MetalCommandList::dispatch(uint32_t x, uint32_t y,
                                     uint32_t z) noexcept {
  TI_ASSERT(current_pipeline_);
  TI_ASSERT(current_shader_resource_set_);

  MTLComputePipelineState_id mtl_compute_pipeline_state =
      current_pipeline_->mtl_compute_pipeline_state();

  NSUInteger local_x = current_pipeline_->workgroup_size().x;
  NSUInteger local_y = current_pipeline_->workgroup_size().y;
  NSUInteger local_z = current_pipeline_->workgroup_size().z;

  std::vector<MetalShaderResource> shader_resources =
      current_shader_resource_set_->resources();

  auto encode_f = [=](MTLCommandBuffer_id mtl_command_buffer) {
    MTLComputeCommandEncoder_id encoder =
        [mtl_command_buffer computeCommandEncoder];

    for (const MetalShaderResource &resource : shader_resources) {
      switch (resource.ty) {
      case MetalShaderResourceType::buffer: {
        [encoder setBuffer:resource.buffer.buffer
                    offset:resource.buffer.offset
                   atIndex:resource.binding];
        break;
      }
      default:
        TI_ASSERT(false);
      }
    }

    [encoder setComputePipelineState:mtl_compute_pipeline_state];
    [encoder dispatchThreadgroups:MTLSizeMake(x, y, z)
            threadsPerThreadgroup:MTLSizeMake(local_x, local_y, local_z)];
    [encoder endEncoding];
  };
  pending_commands_.emplace_back(encode_f);
  return RhiResult::success;
}

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
  *out_cmdlist = new MetalCommandList(*device_);
  return RhiResult::success;
}
StreamSemaphore
MetalStream::submit(CommandList *cmdlist,
                    const std::vector<StreamSemaphore> &wait_semaphores) {
  MetalCommandList *cmdlist2 = (MetalCommandList *)cmdlist;

  @autoreleasepool {
    MTLCommandBuffer_id cmdbuf = [[mtl_command_queue_ commandBuffer] retain];
    for (auto &command : cmdlist2->pending_commands_) {
      command(cmdbuf);
    }
    cmdlist2->pending_commands_.clear();

    [cmdbuf commit];
    pending_cmdbufs_.emplace_back(cmdbuf);
  }

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
  bool family_mac2 = [mtl_device supportsFamily:MTLGPUFamilyMac2];
  bool family_apple7 = [mtl_device supportsFamily:MTLGPUFamilyApple7];
  bool family_apple6 =
      [mtl_device supportsFamily:MTLGPUFamilyApple6] | family_apple7;
  bool family_apple5 =
      [mtl_device supportsFamily:MTLGPUFamilyApple5] | family_apple6;
  bool family_apple4 =
      [mtl_device supportsFamily:MTLGPUFamilyApple4] | family_apple5;
  bool family_apple3 =
      [mtl_device supportsFamily:MTLGPUFamilyApple3] | family_apple4;

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
    caps.set(DeviceCapability::spirv_has_atomic_float, 1);
    caps.set(DeviceCapability::spirv_has_atomic_float_add, 1);
    caps.set(DeviceCapability::spirv_has_atomic_float_minmax, 1);
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

MetalDevice::MetalDevice(MTLDevice_id mtl_device) : mtl_device_(mtl_device) {
  compute_stream_ = std::unique_ptr<MetalStream>(MetalStream::create(*this));

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
    TI_WARN_IF(memory_allocs_.size() != 0,
               "metal device memory leaked: {} unreleased memory allocations",
               memory_allocs_.size());
    memory_allocs_.clear();
    [mtl_device_ release];
    is_destroyed_ = true;
  }
}

DeviceAllocation MetalDevice::allocate_memory(const AllocParams &params) {
  TI_WARN_IF(params.export_sharing, "export sharing is not available in metal");

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

  std::unique_ptr<MetalMemory> memory = std::make_unique<MetalMemory>(buffer);

  DeviceAllocationId alloc_id = (uint64_t)(size_t)memory.get();
  memory_allocs_[alloc_id] = std::move(memory);

  DeviceAllocation out{};
  out.device = this;
  out.alloc_id = alloc_id;
  return out;
}
void MetalDevice::dealloc_memory(DeviceAllocation handle) {
  TI_ASSERT(handle.device == this);
  auto it = memory_allocs_.find(handle.alloc_id);
  memory_allocs_.erase(it);
}
const MetalMemory &MetalDevice::get_memory(DeviceAllocationId alloc_id) const {
  return *memory_allocs_.at(alloc_id);
}

RhiResult MetalDevice::map_range(DevicePtr ptr, uint64_t size,
                                 void **mapped_ptr) {
  const MetalMemory &memory = *memory_allocs_.at(ptr.alloc_id);

  size_t offset = (size_t)ptr.offset;
  TI_ASSERT(offset + size <= memory.size());

  RhiResult result = map(ptr, mapped_ptr);
  *(const uint8_t **)mapped_ptr += offset;
  return result;
}
RhiResult MetalDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  const MetalMemory &memory = *memory_allocs_.at(alloc.alloc_id);
  return memory.mapped_ptr(mapped_ptr);
}
void MetalDevice::unmap(DevicePtr ptr) {}
void MetalDevice::unmap(DeviceAllocation ptr) {}

std::unique_ptr<Pipeline>
MetalDevice::create_pipeline(const PipelineSourceDesc &src, std::string name) {
  TI_ASSERT(src.type == PipelineSourceType::spirv_binary);
  Pipeline *out =
      MetalPipeline::create(*this, (const uint32_t *)src.data, src.size);
  return std::unique_ptr<Pipeline>(out);
}
ShaderResourceSet *MetalDevice::create_resource_set() {
  return new MetalShaderResourceSet(*this);
}

Stream *MetalDevice::get_compute_stream() { return compute_stream_.get(); }
void MetalDevice::wait_idle() { compute_stream_->command_sync(); }

void MetalDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED
}

} // namespace metal
} // namespace taichi::lang
