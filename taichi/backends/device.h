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
  vk_has_atomic_float, // load, store, exchange
  vk_has_atomic_float_add,
  vk_has_atomic_float_minmax,
  vk_has_atomic_float64, // load, store, exchange
  vk_has_atomic_float64_add,
  vk_has_atomic_float64_minmax,
  vk_has_surface,
  vk_has_presentation,
  vk_has_spv_variable_ptr,
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

 private:
  std::unordered_map<DeviceCapability, uint32_t> caps_;
};

}  // namespace lang
}  // namespace taichi