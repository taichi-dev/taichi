// Driver class for source2source kernel codegen. This file is OBSOLETE

#pragma once

#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGen {
 protected:
  Program *prog;
  Kernel *kernel;
  IRNode *ir;

  virtual FunctionType codegen() = 0;

 public:
  KernelCodeGen(Kernel *kernel, IRNode *ir)
      : prog(&kernel->program), kernel(kernel), ir(ir) {
    if (ir == nullptr)
      ir = kernel->ir;
  }

  virtual FunctionType compile();

  static std::unique_ptr<KernelCodeGen> create(Arch arch, Kernel *kernel);

  virtual ~KernelCodeGen() = default;
};

TLANG_NAMESPACE_END