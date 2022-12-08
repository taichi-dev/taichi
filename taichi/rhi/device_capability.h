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
DeviceCapability str2devcap(const std::string_view &name);
const std::string to_string(DeviceCapability c);

// A collection of device capability.
struct DeviceCapabilityConfig {
 public:
  std::map<DeviceCapability, uint32_t> devcaps;

  uint32_t contains(DeviceCapability cap) const;
  uint32_t get(DeviceCapability cap) const;
  void set(DeviceCapability cap, uint32_t level);

  void dbg_print_all() const;

  const std::map<DeviceCapability, uint32_t> &to_inner() const;

  // TI_IO_DEF(devcaps);
};

}  // namespace taichi::lang
