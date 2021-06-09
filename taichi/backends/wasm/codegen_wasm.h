// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

namespace taichi {
namespace lang {

class CodeGenWASM : public KernelCodeGen {
 public:
  CodeGenWASM(Kernel *kernel, IRNode *ir = nullptr) : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;
};

}
}
