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
    runtime_->register_taichi_kernel(name, kernel_data->source_code,
                                     kernel_data->kernel_attribs,
                                     kernel_data->ctx_attribs);
    return std::make_unique<KernelImpl>(runtime_, name);
  }

  KernelManager *const runtime_;
  TaichiAotData aot_data_;
  std::unordered_map<std::string, const CompiledKernelData *> kernels_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params) {
  AotModuleParams params = std::any_cast<AotModuleParams &>(mod_params);
  return std::make_unique<AotModuleImpl>(params);
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
