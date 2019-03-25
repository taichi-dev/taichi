#pragma once

#include "kernel.h"

TLANG_NAMESPACE_BEGIN

class CPUCodeGen : public KernelCodeGen {
 public:
  std::map<int, std::string> masks;

 public:
  CPUCodeGen() : KernelCodeGen() {
    suffix = "cpp";
  }

  void lower() override;

  void codegen() override;
};

TLANG_NAMESPACE_END
