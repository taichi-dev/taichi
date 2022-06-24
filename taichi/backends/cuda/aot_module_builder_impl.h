#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_aot_module_builder.h"

namespace taichi {
namespace lang {
namespace cuda {

class AotModuleBuilderImpl : public LlvmAotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(LlvmProgramImpl *prog)
      : LlvmAotModuleBuilder(prog) {
  }

 private:
  CodeGenLLVM::CompiledData compile_kernel(Kernel *kernel) override;
};

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
