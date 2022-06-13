#pragma once

#include <memory>

#include "taichi/backends/device.h"
#include "taichi/backends/metal/api.h"
#include "taichi/backends/metal/runtime_utils.h"

namespace taichi {
namespace lang {

class MemoryPool;

namespace metal {

struct MTLDevice;

struct ComputeDeviceParams {
  MTLDevice *device{nullptr};
  MemoryPool *mem_pool{nullptr};
  // FIXME: This is a hack since Metal's Device isn't complete yet. Enabling
  // this means that only device allocation functionality is enabled in the
  // Metal Device.
  bool only_for_dev_allocation{false};
};

class AllocToMTLBufferMapper {
 public:
  virtual ~AllocToMTLBufferMapper() = default;

  struct BufferAndMem {
    MTLBuffer *buffer{nullptr};
    BufferMemoryView *mem{nullptr};
  };

  virtual BufferAndMem find(DeviceAllocationId alloc_id) const = 0;

  BufferAndMem find(DeviceAllocation alloc) const {
    return find(alloc.alloc_id);
  }
};

struct MakeDeviceResult {
  std::unique_ptr<taichi::lang::Device> device{nullptr};
  AllocToMTLBufferMapper *mapper{nullptr};
};

MakeDeviceResult make_compute_device(const ComputeDeviceParams &params);

}  // namespace metal
}  // namespace lang
}  // namespace taichi
