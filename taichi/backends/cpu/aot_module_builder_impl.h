#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/llvm/llvm_aot_module_builder.h"

namespace taichi {
namespace lang {
namespace cpu {

class AotModuleBuilderImpl : public LlvmAotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(LlvmProgramImpl *prog)
      : LlvmAotModuleBuilder(prog) {
  }

 private:
  CodeGenLLVM::CompiledData compile_kernel(Kernel *kernel) override;
};

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
