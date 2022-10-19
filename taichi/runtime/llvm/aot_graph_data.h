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
  }

  void launch(RuntimeContext *ctx) override {
    fn_(*ctx);
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
