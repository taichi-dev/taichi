// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#endif

namespace taichi {
namespace lang {

class KernelCodeGenWASM : public KernelCodeGen {
 public:
  KernelCodeGenWASM(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  FunctionType codegen() override;

#ifdef TI_WITH_LLVM
  LLVMCompiledData modulegen(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) override;  // AOT Module Gen
#endif
};

}  // namespace lang
}  // namespace taichi
