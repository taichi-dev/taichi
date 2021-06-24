// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

#include "llvm/IR/Module.h"

namespace taichi {
namespace lang {

class CodeGenWASM : public KernelCodeGen {
 public:
  CodeGenWASM(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;
  std::unique_ptr<llvm::Module> modulegen();
};

}  // namespace lang
}  // namespace taichi
