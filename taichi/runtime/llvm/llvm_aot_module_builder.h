#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  explicit LlvmAotModuleBuilder(const CompileConfig &compile_config,
                                LlvmProgramImpl *prog,
                                TaichiLLVMContext &tlctx)
      : compile_config_(compile_config), prog_(prog), tlctx_(tlctx) {
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

  const CompileConfig &get_compile_config() const {
    return compile_config_;
  }

  TaichiLLVMContext &get_taichi_llvm_context() {
    return tlctx_;
  }

 private:
  mutable LlvmOfflineCache cache_;
  const CompileConfig &compile_config_;
  LlvmProgramImpl *prog_ = nullptr;
  TaichiLLVMContext &tlctx_;
};

}  // namespace taichi::lang
