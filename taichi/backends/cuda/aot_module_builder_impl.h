#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/llvm/llvm_aot_module_builder.h"

namespace taichi {
namespace lang {
namespace cuda {

class AotModuleBuilderImpl : public LlvmAotModuleBuilder {
 private:
  CodeGenLLVM::CompiledData compile_kernel(Kernel *kernel) override;
};

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
