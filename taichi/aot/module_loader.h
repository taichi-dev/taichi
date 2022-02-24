#pragma once

#include <string>
#include <vector>

#include "taichi/aot/module_data.h"
#include "taichi/backends/device.h"
#include "taichi/ir/snode.h"
#include "taichi/aot/module_data.h"

namespace taichi {
namespace lang {

class AotModuleLoader {
 public:
  virtual ~AotModuleLoader() = default;

  // @TODO: Add method get_kernel(...) once the kernel field data will be
  // generic/common across all backends.

  virtual bool get_field(const std::string &name,
                         aot::CompiledFieldData &field) = 0;

  virtual size_t get_root_size() const = 0;
};

// Only responsible for reporting device capabilities
class AotTargetDevice : public Device {
 public:
  AotTargetDevice(Arch arch) {
    // TODO: make this configurable
    set_default_caps(arch);
  }

  void set_default_caps(Arch arch) {
    if (arch == Arch::vulkan) {
      set_cap(DeviceCapability::spirv_version, 0x10300);
    }
  }

  DeviceAllocation allocate_memory(const AllocParams &params) override {
    TI_NOT_IMPLEMENTED;
  }
  void dealloc_memory(DeviceAllocation handle) override {
    TI_NOT_IMPLEMENTED;
  }
  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override {
    TI_NOT_IMPLEMENTED;
  }
  void *map_range(DevicePtr ptr, uint64_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  void *map(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }
  void unmap(DevicePtr ptr) override {
    TI_NOT_IMPLEMENTED;
  }
  void unmap(DeviceAllocation alloc) override {
    TI_NOT_IMPLEMENTED;
  }
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override {
    TI_NOT_IMPLEMENTED;
  }
  Stream *get_compute_stream() override {
    TI_NOT_IMPLEMENTED;
  }
};

}  // namespace lang
}  // namespace taichi
