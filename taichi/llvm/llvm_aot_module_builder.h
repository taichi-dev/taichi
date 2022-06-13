#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/codegen/codegen_llvm.h"

namespace taichi {
namespace lang {

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  virtual CodeGenLLVM::CompiledData compile_kernel(Kernel *kernel) = 0;

 private:
  mutable LlvmOfflineCache cache_;
};

}  // namespace lang
}  // namespace taichi
