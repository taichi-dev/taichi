#pragma once
#include "taichi/util/lang_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"

namespace taichi::lang {
namespace llvm_aot {

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(FunctionType fn,
                      LlvmOfflineCache::KernelCacheData &&kernel_data)
      : kernel_data_(std::move(kernel_data)), fn_(fn) {
    rets = kernel_data_.rets;
    ret_type = kernel_data_.ret_type;
    ret_size = kernel_data_.ret_size;
    nested_parameters.reserve(kernel_data_.args.size());
    for (const auto &kv : kernel_data_.args) {
      nested_parameters[kv.first] = kv.second;
    }
    args_type = kernel_data_.args_type;
    args_size = kernel_data_.args_size;
    arch = Arch::x64;  // Only for letting the launch context builder know
                       // the arch uses LLVM.
                       // TODO: remove arch after the refactoring of
                       //  SPIR-V based backends completes.
    name = kernel_data_.kernel_key;
  }

  void launch(LaunchContextBuilder &ctx) override {
    fn_(ctx);
  }

  LlvmOfflineCache::KernelCacheData kernel_data_;

 private:
  FunctionType fn_;
};

class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(const LlvmOfflineCache::FieldCacheData &field)
      : field_(field) {
  }

  explicit FieldImpl(LlvmOfflineCache::FieldCacheData &&field)
      : field_(std::move(field)) {
  }

  LlvmOfflineCache::FieldCacheData get_snode_tree_cache() const {
    return field_;
  }

 private:
  LlvmOfflineCache::FieldCacheData field_;
};

}  // namespace llvm_aot
}  // namespace taichi::lang
