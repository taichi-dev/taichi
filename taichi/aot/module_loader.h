#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "taichi/aot/module_data.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"
#include "taichi/aot/graph_data.h"

namespace taichi {
namespace lang {

struct RuntimeContext;
class Graph;
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

class TI_DLL_EXPORT KernelTemplateArg {
 public:
  using ArgUnion = std::variant<bool, int64_t, uint64_t, const Field *>;
  template <typename T>
  KernelTemplateArg(const std::string &name, T &&arg)
      : name_(name), targ_(std::forward<T>(arg)) {
  }

 private:
  std::string name_;
  /**
   * @brief Template arg
   *
   */
  ArgUnion targ_;
};

class TI_DLL_EXPORT KernelTemplate {
 public:
  // Rule of 5 to make MSVC happy
  KernelTemplate() = default;
  virtual ~KernelTemplate() = default;
  KernelTemplate(const KernelTemplate &) = delete;
  KernelTemplate &operator=(const KernelTemplate &) = delete;
  KernelTemplate(KernelTemplate &&) = default;
  KernelTemplate &operator=(KernelTemplate &&) = default;

  Kernel *get_kernel(const std::vector<KernelTemplateArg> &template_args);

 protected:
  virtual std::unique_ptr<Kernel> make_new_kernel(
      const std::vector<KernelTemplateArg> &template_args) = 0;

 private:
  std::unordered_map<std::string, std::unique_ptr<Kernel>> loaded_kernels_;
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

  static std::unique_ptr<Module> load(Arch arch, std::any mod_params);

  // Module metadata
  // TODO: Instead of virtualize these simple properties, just store them as
  // member variables.
  virtual Arch arch() const = 0;
  virtual uint64_t version() const = 0;
  virtual size_t get_root_size() const = 0;

  Kernel *get_kernel(const std::string &name);
  KernelTemplate *get_kernel_template(const std::string &name);
  Field *get_snode_tree(const std::string &name);

  virtual std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) {
    TI_NOT_IMPLEMENTED;
  }

  inline bool is_corrupted() const {
    return is_corrupted_;
  }

 protected:
  virtual std::unique_ptr<Kernel> make_new_kernel(const std::string &name) = 0;
  virtual std::unique_ptr<KernelTemplate> make_new_kernel_template(
      const std::string &name) = 0;
  virtual std::unique_ptr<Field> make_new_field(const std::string &name) = 0;
  inline void mark_corrupted() {
    is_corrupted_ = true;
  }
  std::unordered_map<std::string, CompiledGraph> graphs_;

 private:
  bool is_corrupted_{false};
  std::unordered_map<std::string, std::unique_ptr<Kernel>> loaded_kernels_;
  std::unordered_map<std::string, std::unique_ptr<KernelTemplate>>
      loaded_kernel_templates_;
  std::unordered_map<std::string, std::unique_ptr<Field>> loaded_fields_;
};

// Only responsible for reporting device capabilities
class TargetDevice : public Device {
 public:
  TargetDevice(Arch arch) {
    // TODO: make this configurable
    set_default_caps(arch);
  }

  void set_default_caps(Arch arch) {
    if (arch == Arch::vulkan || arch == Arch::opengl) {
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
  void wait_idle() override {
    TI_NOT_IMPLEMENTED;
  }
};

}  // namespace aot
}  // namespace lang
}  // namespace taichi
