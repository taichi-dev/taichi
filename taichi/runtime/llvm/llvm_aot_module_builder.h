#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  explicit LlvmAotModuleBuilder(KernelCompilationManager &compilation_manager,
                                const CompileConfig &compile_config,
                                LlvmProgramImpl *prog)
      : compilation_manager_(compilation_manager),
        compile_config_(compile_config),
        prog_(prog) {
  }

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

 private:
  LLVMCompiledKernel compile_kernel(Kernel *kernel);

  mutable LlvmOfflineCache cache_;
  KernelCompilationManager &compilation_manager_;
  const CompileConfig &compile_config_;
  LlvmProgramImpl *prog_ = nullptr;
};

}  // namespace taichi::lang
