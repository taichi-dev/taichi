#pragma once
#include "taichi/lang_util.h"

#include "taichi/program/compile_config.h"

namespace taichi {
namespace lang {

// For backend dependent code (e.g. codegen)
// Or the backend runtime itself
// Capabilities are per-device
enum class VulkanCapabilities : uint32_t {
  API_VERSION,
  SPIRV_VERSION,
  I64_SUPPORT,
  I64_ATOMIC_SUPPORT,
  F64_SUPPORT,
  F64_ATOMIC_SUPPORT,
  I16_SUPPORT,
  F16_SUPPORT,
  I8_SUPPORT,
  VARIABLE_PTR_SUPPORT,
  PHYSICAL_ADDR_SUPPORT,
  MEM_EXPORT_SUPPORT,
  SPARSE_MEMORY_SUPPORT
};

class Device {
 public:
  virtual ~Device(){};

  virtual uint32_t getCapability(uint32_t capability_id) const {
    return caps_.at(capability_id);
  }

  virtual uint32_t setCapability(uint32_t capability_id, uint32_t val) {
    caps_[capability_id] = val;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> caps_;
};

class VulkanDevice : public Device {
 public:
  VulkanDevice();
  ~VulkanDevice();
};

}  // namespace lang
}  // namespace taichi