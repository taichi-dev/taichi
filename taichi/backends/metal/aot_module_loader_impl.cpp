#include "taichi/backends/metal/aot_module_loader_impl.h"

#include "taichi/backends/metal/aot_utils.h"
#include "taichi/backends/metal/kernel_manager.h"

namespace taichi {
namespace lang {
namespace metal {
namespace {

class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(KernelManager *runtime, const CompiledFieldData &field)
      : runtime_(runtime), field_(field) {
  }

 private:
  KernelManager *const runtime_;
  CompiledFieldData field_;
};

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(KernelManager *runtime, const std::string &kernel_name)
      : runtime_(runtime), kernel_name_(kernel_name) {
  }

  void launch(RuntimeContext *ctx) override {
    runtime_->launch_taichi_kernel(kernel_name_, ctx);
  }

 private:
  KernelManager *const runtime_;
  const std::string kernel_name_;
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params)
      : runtime_(params.runtime) {
    const std::string bin_path =
        fmt::format("{}/metadata.tcb", params.module_path);
    read_from_binary_file(aot_data_, bin_path);
    // Do we still need to load each individual kernel?
    for (const auto &k : aot_data_.kernels) {
      kernels_[k.kernel_name] = &k;
    }
    for (const auto &f : aot_data_.fields) {
      fields_[f.field_name] = &f;
    }
  }

  size_t get_root_size() const override {
    return aot_data_.metadata.root_buffer_size;
  }

  // Module metadata
  Arch arch() const override {
    return Arch::metal;
  }
  uint64_t version() const override {
    TI_NOT_IMPLEMENTED;
  }

 private:
  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    auto itr = kernels_.find(name);
    if (itr == kernels_.end()) {
      TI_DEBUG("Failed to load kernel {}", name);
      return nullptr;
    }
    auto *kernel_data = itr->second;
    runtime_->register_taichi_kernel(
        name, kernel_data->source_code, kernel_data->kernel_attribs,
        kernel_data->ctx_attribs, /*kernel=*/nullptr);
    return std::make_unique<KernelImpl>(runtime_, name);
  }

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override {
    auto itr = fields_.find(name);
    if (itr == fields_.end()) {
      TI_DEBUG("Failed to load field {}", name);
      return nullptr;
    }
    auto *field_data = itr->second;
    return std::make_unique<FieldImpl>(runtime_, *field_data);
  }

  KernelManager *const runtime_;
  TaichiAotData aot_data_;
  std::unordered_map<std::string, const CompiledKernelData *> kernels_;
  std::unordered_map<std::string, const CompiledFieldData *> fields_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
