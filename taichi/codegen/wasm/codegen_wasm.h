// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#endif

namespace taichi::lang {

class KernelCodeGenWASM : public KernelCodeGen {
 public:
  explicit KernelCodeGenWASM(const CompileConfig &compile_config,
                             Kernel *kernel)
      : KernelCodeGen(compile_config, kernel) {
  }

  FunctionType compile_to_function() override;

#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) override;  // AOT Module Gen

  LLVMCompiledKernel compile_kernel_to_module() override;
#endif
};

}  // namespace taichi::lang
