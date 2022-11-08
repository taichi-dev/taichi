#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  explicit LlvmAotModuleBuilder(LlvmProgramImpl *prog) : prog_(prog) {
  }

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  virtual LLVMCompiledKernel compile_kernel(Kernel *kernel) = 0;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

  const LlvmOfflineCache &get_cache() {
    return cache_;
  }

 private:
  mutable LlvmOfflineCache cache_;
  LlvmProgramImpl *prog_ = nullptr;
};

}  // namespace taichi::lang
