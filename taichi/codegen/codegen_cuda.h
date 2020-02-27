// The CUDA backend

#pragma once

#include "kernel.h"

TLANG_NAMESPACE_BEGIN

class GPUCodeGen : public KernelCodeGen {
 public:
 public:
  GPUCodeGen(const std::string &kernel_name) : KernelCodeGen(kernel_name) {
  }

  void lower_llvm();

  void lower() override;

  virtual FunctionType codegen_llvm() override;
};

TLANG_NAMESPACE_END
