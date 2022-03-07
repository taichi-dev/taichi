#pragma once

#include <any>
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

namespace aot {

class TI_DLL_EXPORT Field {
 public:
  // Rule of 5 to make MSVC happy
  Field() = default;
  virtual ~Field() = default;
  Field(const Field &) = delete;
  Field &operator=(const Field &) = delete;
  Field(Field &&) = default;
  Field &operator=(Field &&) = default;
};

class TI_DLL_EXPORT Kernel {
 public:
  // Rule of 5 to make MSVC happy
  Kernel() = default;
  virtual ~Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(RuntimeContext *ctx) = 0;
};

class TI_DLL_EXPORT Module {
 public:
  // Rule of 5 to make MSVC happy
  Module() = default;
  virtual ~Module() = default;
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&) = default;
  Module &operator=(Module &&) = default;

  static std::unique_ptr<Module> load(const std::string &path,
                                      Arch arch,
                                      std::any mod_params);

  // Module metadata
  Arch arch() const;
  uint64_t version() const;

  /**
   * Intended to be overriden by each backend's implementation.
   */
  virtual std::unique_ptr<Kernel> get_kernel(const std::string &name) = 0;
  virtual std::unique_ptr<Field> get_field(const std::string &name) = 0;
  virtual size_t get_root_size() const = 0;

 protected:
  virtual std::unique_ptr<Kernel> make_new_kernel(const std::string &name) = 0;

 private:
  std::unordered_map<std::string, std::unique_ptr<Kernel>> loaded_kernels_;
};

// Only responsible for reporting device capabilities
class TargetDevice : public Device {
 public:
  TargetDevice(Arch arch) {
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

}  // namespace aot
}  // namespace lang
}  // namespace taichi
