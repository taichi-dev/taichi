#pragma once
#include <memory>
#include "taichi/platform/mac/objc_api.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/metal/metal_api.h"

namespace taichi::lang {

namespace metal {

struct MetalDevice : public Device {
 public:
  explicit MetalDevice(mac::nsobj_unique_ptr<MTL::Device>&& device);

  static std::unique_ptr<MetalDevice> create();

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;
  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

  std::unique_ptr<Pipeline> create_pipeline(const PipelineSourceDesc &src,
                                            std::string name) override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  Stream *get_compute_stream() override;
  void wait_idle() override;

  void import_mtl_buffer(MTL::Buffer* buffer);
  MTL::Buffer* get_mtl_buffer(DeviceAllocationId alloc_id) const;

  constexpr MTL::Device* get_mtl_device() const {
    return device_.get();
  }

 private:
  mac::nsobj_unique_ptr<MTL::Device> device_;

  struct ThreadLocalStreams {
    std::unordered_map<std::thread::id, std::unique_ptr<MetalStream>> map;
  };
  std::unique_ptr<ThreadLocalStreams> compute_stream_;

  struct AllocationInternal {
    bool external{false};
    mac::nsobj_unique_ptr<MTL::Buffer> buffer{nullptr};
  };

  std::unordered_map<DeviceAllocationId, AllocationInternal> allocations_;
  DeviceAllocationId next_alloc_id_{0};
};

}  // namespace metal
}  // namespace taichi::lang
