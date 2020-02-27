// The CUDA backend

#pragma once

#include "codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCUDA : public KernelCodeGen {
 public:
  CodeGenCUDA(Kernel *kernel) : KernelCodeGen(kernel) {
  }

  void lower() override;

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
