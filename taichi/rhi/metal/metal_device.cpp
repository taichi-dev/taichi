#include "taichi/rhi/metal/metal_device.h"
#include "taichi/rhi/metal/metal_pipeline.h"
#include "taichi/rhi/metal/metal_stream.h"

namespace taichi::lang {
namespace metal {

MetalDevice::MetalDevice(mac::nsobj_unique_ptr<MTL::Device>&& device) :
  device_(std::move(device)),
  compute_stream_(std::make_unique<ThreadLocalStreams>())
{
  TI_ERROR_UNLESS(device_ != nullptr, "invalid device");
}
std::unique_ptr<MetalDevice> MetalDevice::create() {
  mac::nsobj_unique_ptr<MTL::Device> device =
      mac::wrap_as_nsobj_unique_ptr(MTL::CreateSystemDefaultDevice());
  if (device == nullptr) {
    return nullptr;
  }

  return std::make_unique<MetalDevice>(std::move(device));
}

DeviceAllocation MetalDevice::allocate_memory(const AllocParams &params) {
  DeviceAllocation res {};
  res.device = this;
  // Do not use `allocations_.size()` as `alloc_id`, as items could be erased
  // from `allocations_`.
  res.alloc_id = next_alloc_id_++;

  // The created MTLBuffer has its storege mode being .manged.
  // API ref:
  // https://developer.apple.com/documentation/metal/mtldevice/1433382-makebuffer
  //
  // We initially used .shared storage mode, meaning the GPU and CPU shared
  // the system memory. This turned out to be slow as page fault on GPU was
  // very costly. By switching to .managed mode, on GPUs with discrete memory
  // model, the data will reside in both GPU's VRAM and CPU's system RAM. This
  // made the GPU memory access much faster. But we will need to manually
  // synchronize the buffer resources between CPU and GPU.
  //
  // See also:
  // https://developer.apple.com/documentation/metal/synchronizing_a_managed_resource
  // https://developer.apple.com/documentation/metal/setting_resource_storage_modes/choosing_a_resource_storage_mode_in_macos
  AllocationInternal alloc {};
  alloc.external = false;
  alloc.buffer = mac::wrap_as_nsobj_unique_ptr(device_->newBuffer(
      params.size,
      MTL::ResourceCPUCacheModeDefaultCache | MTL::StorageModeManaged));
  allocations_[res.alloc_id] = std::move(alloc);
  return res;
}
void MetalDevice::dealloc_memory(DeviceAllocation handle) {
  allocations_.erase(handle.alloc_id);
}

void *MetalDevice::map_range(DevicePtr ptr, uint64_t size) {
  auto it = allocations_.find(ptr.alloc_id);
  if (it == allocations_.end()) {
    return nullptr;
  }
  if ((ptr.offset + size) > it->second.buffer->allocatedSize()) {
    TI_ERROR("Range exceeded");
    return nullptr;
  }
  return (uint8_t*)it->second.buffer->contents() + ptr.offset;
}
void *MetalDevice::map(DeviceAllocation alloc) {
  auto it = allocations_.find(alloc.alloc_id);
  if (it == allocations_.end()) {
    return nullptr;
  }
  return it->second.buffer->contents();
}
void MetalDevice::unmap(DevicePtr ptr) {
  // No-op on Metal
}
void MetalDevice::unmap(DeviceAllocation alloc) {
  // No-op on Metal
}


std::unique_ptr<Pipeline> MetalDevice::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  return MetalPipeline::create(this, src, name);
}

void MetalDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  Stream *stream = get_compute_stream();
  std::unique_ptr<CommandList> cmd = stream->new_command_list();
  cmd->buffer_copy(dst, src, size);
  stream->submit_synced(cmd.get());
}

Stream *MetalDevice::get_compute_stream() {
  auto& streams = compute_stream_->map[std::this_thread::get_id()];
  if (streams == nullptr) {
    streams = MetalStream::create(this);
  }
  return streams.get();
}
void MetalDevice::wait_idle() {
  for (auto& pair : compute_stream_->map) {
    pair.second->command_sync();
  }
}

void MetalDevice::import_mtl_buffer(MTL::Buffer* buffer) {
  AllocationInternal alloc {};
  alloc.external = true;
  // (penguinliong) Retain to ensure the buffer is not released when destroyed
  // by Taichi.
  alloc.buffer = mac::retain_and_wrap_as_nsobj_unique_ptr(buffer);
  allocations_[next_alloc_id_++] = std::move(alloc);
}
MTL::Buffer* MetalDevice::get_mtl_buffer(DeviceAllocationId alloc_id) const {
  auto it = allocations_.find(alloc_id);
  if (it == allocations_.end()) {
    return nullptr;
  } else {
    return it->second.buffer.get();
  }
}

}  // namespace metal
}  // namespace taichi::lang
