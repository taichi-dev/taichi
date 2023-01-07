#include "taichi/rhi/metal/metal_device.h"
#include "taichi/common/logging.h"
#include "taichi/rhi/device.h"
#include "taichi/runtime/metal/api.h"
#include <functional>
#include <memory>

namespace taichi::lang {
namespace metal {

MetalMemory::MetalMemory(MTLBuffer_id mtl_buffer)
    : mtl_buffer_(mtl_buffer) {}
MetalMemory::~MetalMemory() { [mtl_buffer_ release]; }

MTLBuffer_id MetalMemory::mtl_buffer() const {
  return mtl_buffer_;
}
size_t MetalMemory::size() const {
  return (size_t)[mtl_buffer_ length];
}
RhiResult MetalMemory::mapped_ptr(void **mapped_ptr) const {
  void *ptr = [mtl_buffer_ contents];
  if (ptr == nullptr) {
    return RhiResult::invalid_usage;
  } else {
    *mapped_ptr = ptr;
    return RhiResult::success;
  }
}


MetalCommandList::MetalCommandList(const MetalDevice &device) : device_(&device) {
}
MetalCommandList::~MetalCommandList() {
}

void MetalCommandList::memory_barrier() {
  // Note that resources created from `MTLDevice` (which is the only available
  // way to allocate resource here) are `MTLHazardTrackingModeTracked` by
  // default. So we don't have to barrier explicitly.
}

void MetalCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  const MetalMemory& src_memory = device_->get_memory(src.alloc_id);
  const MetalMemory& dst_memory = device_->get_memory(dst.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size_t src_size = src_memory.size();
    size_t dst_size = dst_memory.size();
    TI_ASSERT(src_size == dst_size);
    size = src_size;
  }

  MTLBuffer_id src_mtl_buffer = src_memory.mtl_buffer();
  MTLBuffer_id dst_mtl_buffer = dst_memory.mtl_buffer();

  std::function<void(MTLCommandBuffer_id)> encode_f = [=](MTLCommandBuffer_id mtl_command_buffer) {
    MTLBlitCommandEncoder_id encoder = [mtl_command_buffer blitCommandEncoder];
    [encoder copyFromBuffer:src_mtl_buffer sourceOffset:(NSUInteger)src.offset toBuffer:dst_mtl_buffer destinationOffset:(NSUInteger)dst.offset size:size];
    [encoder endEncoding];
  };
  pending_commands_.emplace_back(encode_f);
}
void MetalCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  TI_ASSERT(data == 0);

  const MetalMemory& memory = device_->get_memory(ptr.alloc_id);

  if (size == kBufferSizeEntireSize) {
    size = memory.size();
  }

  MTLBuffer_id mtl_buffer = memory.mtl_buffer();

  std::function<void(MTLCommandBuffer_id)> encode_f = [=](MTLCommandBuffer_id mtl_command_buffer) {
    MTLBlitCommandEncoder_id encoder = [mtl_command_buffer blitCommandEncoder];
    [encoder fillBuffer:mtl_buffer range:NSMakeRange((NSUInteger)ptr.offset, (NSUInteger)size) value:0];
    [encoder endEncoding];
  };
  pending_commands_.emplace_back(encode_f);
}


MetalStream::MetalStream(const MetalDevice& device, MTLCommandQueue_id mtl_command_queue) : device_(&device), mtl_command_queue_(mtl_command_queue) {}
MetalStream::~MetalStream() { [mtl_command_queue_ release]; }

std::unique_ptr<CommandList> MetalStream::new_command_list() {
  return std::unique_ptr<CommandList>(new MetalCommandList(*device_));
}
StreamSemaphore MetalStream::submit(
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &wait_semaphores) {
  MetalCommandList* cmdlist2 = (MetalCommandList*)cmdlist;

  @autoreleasepool
  {
    MTLCommandBuffer_id cmdbuf = [[mtl_command_queue_ commandBuffer] retain];
    for (auto& command : cmdlist2->pending_commands_) {
      command(cmdbuf);
    }
    cmdlist2->pending_commands_.clear();

    [cmdbuf commit];
    pending_cmdbufs_.emplace_back(cmdbuf);
  }

  return {};
}
StreamSemaphore MetalStream::submit_synced(CommandList *cmdlist, const std::vector<StreamSemaphore> &wait_semaphores) {
  auto sema = submit(cmdlist, wait_semaphores);
  command_sync();
  return sema;
}
void MetalStream::command_sync() {
  for (const auto& cmdbuf : pending_cmdbufs_) {
    [cmdbuf waitUntilCompleted];
    [cmdbuf release];
  }
  pending_cmdbufs_.clear();
}



MetalDevice::MetalDevice(MTLDevice_id mtl_device) : mtl_device_(mtl_device) {
  MTLCommandQueue_id compute_queue = [mtl_device newCommandQueue];
  compute_stream_ = std::make_unique<MetalStream>(*this, compute_queue);
}
MetalDevice::~MetalDevice() { destroy(); }

std::unique_ptr<MetalDevice> MetalDevice::create() {
  MTLDevice_id mtl_device = MTLCreateSystemDefaultDevice();

  std::unique_ptr<MetalDevice> out = std::make_unique<MetalDevice>(mtl_device);
  return out;
}
void MetalDevice::destroy() {
  is_destroyed_ = true;
  TI_WARN_IF(memory_allocs_.size() != 0,
             "metal device memory leaked: {} unreleased memory allocations",
             memory_allocs_.size());
  [mtl_device_ release];
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

  MTLBuffer_id buffer =
      [mtl_device_ newBufferWithLength:params.size
                               options:resource_options]; // retain

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
void MetalDevice::unmap(DevicePtr ptr) {
}
void MetalDevice::unmap(DeviceAllocation ptr) {
}


std::unique_ptr<Pipeline>
MetalDevice::create_pipeline(const PipelineSourceDesc &src,
                             std::string name){TI_NOT_IMPLEMENTED}

Stream *MetalDevice::get_compute_stream() {
  return compute_stream_.get();
}
void MetalDevice::wait_idle() {
  compute_stream_->command_sync();
}

ShaderResourceSet *MetalDevice::create_resource_set() {
  TI_NOT_IMPLEMENTED
}

void MetalDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED
}


} // namespace metal
} // namespace taichi::lang
