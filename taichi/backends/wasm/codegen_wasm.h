// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenWASM : public KernelCodeGen {
 public:
  CodeGenWASM(Kernel *kernel, IRNode *ir = nullptr) : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
