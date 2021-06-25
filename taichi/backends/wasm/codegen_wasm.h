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
};

class CodeGenWASMAOT : public CodeGenWASM {
 public:
  CodeGenWASMAOT(Kernel *kernel, IRNode *ir = nullptr, 
              std::unique_ptr<llvm::Module> &&module = nullptr)
      : CodeGenWASM(kernel, ir),
        module(std::move(module)) {
  }

  std::pair<std::unique_ptr<llvm::Module>,
            std::unique_ptr<std::vector<std::string>>> modulegen();

 private:
  std::unique_ptr<llvm::Module> module;
};

}  // namespace lang
}  // namespace taichi
