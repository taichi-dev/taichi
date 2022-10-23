#pragma once
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"

namespace taichi::lang::metal {

struct MetalResourceBinder : public ResourceBinder {
public:
  struct Binding {
    MTL::Buffer *buffer;
    uint64_t offset;
    uint64_t size;

    Binding() = default;
    Binding(MTL::Buffer *buffer, uint64_t offset, uint64_t size) :
      buffer(buffer), offset(offset), size(size) {}
  };

  explicit MetalResourceBinder(MetalDevice* device);

  std::unique_ptr<Bindings> materialize() override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override;

  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override;
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override;

  constexpr const std::map<uint32_t, Binding>& get_bindings() const {
    return bindings_;
  }

private:
  friend struct MetalCommandList;

  MetalDevice* device_;
  std::map<uint32_t, Binding> bindings_;
};

}  // namespace taichi::lang::metal
