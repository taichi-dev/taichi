#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/aot/module_data.h"
#include "taichi/backends/device.h"
#include "taichi/ir/snode.h"
#include "taichi/aot/module_data.h"

namespace taichi {
namespace lang {

class RuntimeContext;

// TODO: Instead of prefixing all these classes with "Aot", just put them into
// the `aot` namespace.
class TI_DLL_EXPORT AotKernel {
 public:
  // Rule of 5 to make MSVC happy
  AotKernel() = default;
  virtual ~AotKernel() = default;
  AotKernel(const AotKernel &) = delete;
  AotKernel &operator=(const AotKernel &) = delete;
  AotKernel(AotKernel &&) = default;
  AotKernel &operator=(AotKernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(RuntimeContext *ctx) = 0;
};

class TI_DLL_EXPORT AotModuleLoader {
 public:
  // Rule of 5 to make MSVC happy
  AotModuleLoader() = default;
  virtual ~AotModuleLoader() = default;
  AotModuleLoader(const AotModuleLoader &) = delete;
  AotModuleLoader &operator=(const AotModuleLoader &) = delete;
  AotModuleLoader(AotModuleLoader &&) = default;
  AotModuleLoader &operator=(AotModuleLoader &&) = default;

  // TODO: Add method get_kernel(...) once the kernel field data will be
  // generic/common across all backends.

  virtual bool get_field(const std::string &name,
                         aot::CompiledFieldData &field) = 0;

  /**
   * @brief Get the kernel object
   *
   * @param name Name of the kernel
   * @return AotKernel*
   */
  AotKernel *get_kernel(const std::string &name);

  virtual size_t get_root_size() const = 0;

 protected:
  virtual std::unique_ptr<AotKernel> make_new_kernel(
      const std::string &name) = 0;

 private:
  std::unordered_map<std::string, std::unique_ptr<AotKernel>> loaded_kernels_;
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
