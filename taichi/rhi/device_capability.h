#pragma once
#include <string>
#include <map>
#include <cstdint>

namespace taichi::lang {

// For backend dependent code (e.g. codegen)
// Or the backend runtime itself
// Capabilities are per-device
enum class DeviceCapability : uint32_t {
#define PER_DEVICE_CAPABILITY(name) name,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_DEVICE_CAPABILITY
};
const std::string to_string(DeviceCapability c);

// A collection of device capability.
struct DeviceCapabilityConfig {
 private:
  std::map<DeviceCapability, uint32_t> inner_;

 public:
  inline uint32_t get(DeviceCapability cap) const {
    auto it = inner_.find(cap);
    if (it != inner_.end()) {
      return it->second;
    }
    return 0;
  }
  inline void set(DeviceCapability cap, uint32_t level) {
    inner_[cap] = level;
  }
  inline void dbg_print_all() const {
    for (auto &pair : inner_) {
      TI_TRACE("DeviceCapability::{} ({}) = {}", to_string(pair.first),
               int(pair.first), pair.second);
    }
  }
  inline const std::map<DeviceCapability, uint32_t> &to_inner() const {
    return inner_;
  }
};

}  // namespace taichi::lang
