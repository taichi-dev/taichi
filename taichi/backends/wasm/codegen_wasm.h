// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

#include "llvm/IR/Module.h"

namespace taichi {
namespace lang {

class ModuleGenValue {
 public:
  ModuleGenValue(std::unique_ptr<llvm::Module> module,
                 const std::vector<std::string> &name_list)
      : module(std::move(module)), name_list(name_list) {
  }
  std::unique_ptr<llvm::Module> module;
  std::vector<std::string> name_list;
};

class CodeGenWASM : public KernelCodeGen {
 public:
  CodeGenWASM(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;

  std::unique_ptr<ModuleGenValue> modulegen(
      std::unique_ptr<llvm::Module> &&module);  // AOT Module Gen
};

}  // namespace lang
}  // namespace taichi
