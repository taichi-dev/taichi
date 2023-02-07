#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

namespace taichi::lang {
namespace cpu {

class AotModuleBuilderImpl : public LlvmAotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(const CompileConfig &compile_config,
                                LlvmProgramImpl *prog,
                                TaichiLLVMContext &tlctx)
      : LlvmAotModuleBuilder(compile_config, prog, tlctx) {
  }

 private:
  LLVMCompiledKernel compile_kernel(Kernel *kernel) override;
};

}  // namespace cpu
}  // namespace taichi::lang
