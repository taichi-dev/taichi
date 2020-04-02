// x86 backend implementation

#pragma once

#include "codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCPU : public KernelCodeGen {
 public:
  CodeGenCPU(Kernel *kernel) : KernelCodeGen(kernel) {
  }

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
