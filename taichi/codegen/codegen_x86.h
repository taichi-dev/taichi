// x86 backend implementation

#pragma once

#include "kernel.h"

TLANG_NAMESPACE_BEGIN

class CPUCodeGen : public KernelCodeGen {
 public:
  CPUCodeGen(const std::string &kernel_name) : KernelCodeGen(kernel_name) {
  }

  void lower() override;

  virtual FunctionType codegen_llvm() override;
};

TLANG_NAMESPACE_END
