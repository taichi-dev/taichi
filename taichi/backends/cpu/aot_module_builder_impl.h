#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/llvm/llvm_offline_cache.h"

namespace taichi {
namespace lang {
namespace cpu {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

 private:
  mutable LlvmOfflineCache cache_;
};

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
