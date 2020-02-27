// x86 backend implementation

#pragma once

#include "codegen.h"

TLANG_NAMESPACE_BEGIN

class CPUCodeGen : public KernelCodeGen {
 public:
  CPUCodeGen(Kernel *kernel) : KernelCodeGen(kernel) {
  }

  void lower() override;

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
