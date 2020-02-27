// The CUDA backend

#pragma once

#include "codegen.h"

TLANG_NAMESPACE_BEGIN

class GPUCodeGen : public KernelCodeGen {
 public:
 public:
  GPUCodeGen(Kernel *kernel) : KernelCodeGen(kernel) {
  }

  void lower() override;

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
