#pragma once
#include "taichi/lang_util.h"

#include "taichi/program/compile_config.h"

namespace taichi {
namespace lang {

// For backend dependent code (e.g. codegen)
// Or the backend runtime itself
// Capabilities are per-device
enum class DeviceCapability : uint32_t {
  vk_api_version,
  vk_spirv_version,
  vk_has_physical_features2,
  vk_has_int8,
  vk_has_int16,
  vk_has_int64,
  vk_has_float16,
  vk_has_float64,
  vk_has_external_memory,
  vk_has_atomic_i64,
  vk_has_atomic_float,  // load, store, exchange
  vk_has_atomic_float_add,
  vk_has_atomic_float_minmax,
  vk_has_atomic_float64,  // load, store, exchange
  vk_has_atomic_float64_add,
  vk_has_atomic_float64_minmax,
  vk_has_surface,
  vk_has_presentation,
  vk_has_spv_variable_ptr,
};

class Device;
struct DeviceAllocation;
struct DevicePtr;

struct DeviceAllocation {
  Device *device{nullptr};
  uint32_t alloc_id{0};

  DevicePtr get_ptr(uint64_t offset) const;
};

struct DeviceAllocationUnique : public DeviceAllocation {
  DeviceAllocationUnique(DeviceAllocation alloc) : DeviceAllocation(alloc) {}
  ~DeviceAllocationUnique();
};

struct DevicePtr {
  const DeviceAllocation *allocation{nullptr};
  uint64_t offset{0};
};

class Device {
 public:
  virtual ~Device(){};

  virtual uint32_t get_cap(DeviceCapability capability_id) const {
    if (caps_.find(capability_id) == caps_.end())
      return 0;
    return caps_.at(capability_id);
  }

  virtual void set_cap(DeviceCapability capability_id, uint32_t val) {
    caps_[capability_id] = val;
  }

  struct AllocParams {
    uint64_t size{0};
    bool host_read{false};
    bool host_write{false};
  };

  virtual DeviceAllocation allocate_memory(const AllocParams& params) = 0;
  virtual void dealloc_memory(DeviceAllocation allocation) = 0;

  std::unique_ptr<DeviceAllocationUnique> allocate_memory_unique(const AllocParams& params) {
    return std::make_unique<DeviceAllocationUnique>(this->allocate_memory(params));
  }

  // Mapping can fail and will return nullptr
  virtual void* map_range(DevicePtr ptr, uint64_t size) = 0;
  virtual void* map(DeviceAllocation alloc) = 0;

  virtual void unmap(DevicePtr ptr) = 0;
  virtual void unmap(DeviceAllocation alloc) = 0;

  // Directly share memory in the form of alias
  static DeviceAllocation share_to(DeviceAllocation *alloc, Device *target);
  
  // Strictly intra device copy
  virtual void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) = 0;

  // Copy memory inter or intra devices
  static void memcpy(DevicePtr dst, DevicePtr src, uint64_t size);

 private:
  std::unordered_map<DeviceCapability, uint32_t> caps_;
};

}  // namespace lang
}  // namespace taichi
