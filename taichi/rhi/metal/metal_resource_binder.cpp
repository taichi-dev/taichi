#include "taichi/rhi/metal/metal_resource_binder.h"
#include "taichi/rhi/metal/metal_device.h"

namespace taichi::lang::metal {

MetalResourceBinder::MetalResourceBinder(MetalDevice* device) : device_(device) {}

void MetalResourceBinder::rw_buffer(uint32_t set,
                uint32_t binding,
                DevicePtr ptr,
                size_t size) {
  MTL::Buffer* buffer = device_->get_mtl_buffer(ptr.alloc_id);
  TI_ASSERT(buffer != nullptr);
  bindings_[binding] = Binding(buffer, ptr.offset, size);
}
void MetalResourceBinder::rw_buffer(uint32_t set,
                uint32_t binding,
                DeviceAllocation alloc) {
  MTL::Buffer* buffer = device_->get_mtl_buffer(alloc.alloc_id);
  TI_ASSERT(buffer != nullptr);
  bindings_[binding] = Binding(buffer, 0, buffer->allocatedSize());
}

void MetalResourceBinder::buffer(uint32_t set,
            uint32_t binding,
            DevicePtr ptr,
            size_t size) {
  MTL::Buffer* buffer = device_->get_mtl_buffer(ptr.alloc_id);
  TI_ASSERT(buffer != nullptr);
  bindings_[binding] = Binding(buffer, ptr.offset, size);
}
void MetalResourceBinder::buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) {
  MTL::Buffer* buffer = device_->get_mtl_buffer(alloc.alloc_id);
  TI_ASSERT(buffer != nullptr);
  bindings_[binding] = Binding(buffer, 0, buffer->allocatedSize());
}

} // namespace taichi::lang::metal
