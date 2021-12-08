// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#endif

namespace taichi {
namespace lang {

#ifdef TI_WITH_LLVM
class ModuleGenValue {
 public:
  ModuleGenValue(std::unique_ptr<llvm::Module> module,
                 const std::vector<std::string> &name_list)
      : module(std::move(module)), name_list(name_list) {
  }
  std::unique_ptr<llvm::Module> module;
  std::vector<std::string> name_list;
};
#endif

class CodeGenWASM : public KernelCodeGen {
 public:
  CodeGenWASM(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  FunctionType codegen() override;

#ifdef TI_WITH_LLVM
  std::unique_ptr<ModuleGenValue> modulegen(
      std::unique_ptr<llvm::Module> &&module);  // AOT Module Gen
#endif
};

}  // namespace lang
}  // namespace taichi
