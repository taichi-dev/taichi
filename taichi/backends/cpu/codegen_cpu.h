// x86 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCPU : public KernelCodeGen {
 public:
  CodeGenCPU(Kernel *kernel, IRNode *ir = nullptr) : KernelCodeGen(kernel, ir) {
  }

  static std::unique_ptr<CodeGenLLVM> make_codegen_llvm(Kernel *kernel,
                                                        IRNode *ir);

  FunctionType codegen() override;
};

TLANG_NAMESPACE_END
