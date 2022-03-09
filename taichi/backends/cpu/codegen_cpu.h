// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCPU : public KernelCodeGen {
 public:
  CodeGenCPU(Kernel *kernel, IRNode *ir = nullptr, bool needs_cache = false)
      : KernelCodeGen(kernel, ir), needs_cache_(needs_cache) {
  }

  FunctionType codegen() override;

 private:
  bool needs_cache_{false};
};

TLANG_NAMESPACE_END
