#pragma once

#include "kernel.h"

TLANG_NAMESPACE_BEGIN

class GPUCodeGen : public KernelCodeGen {
 public:
 public:
  GPUCodeGen(const std::string &kernel_name) : KernelCodeGen(kernel_name) {
    suffix = "cu";
  }

  void lower() override;

  void codegen() override;
};

TLANG_NAMESPACE_END
