// Driver class for source2source kernel codegen. This file is OBSOLETE

#pragma once

#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGen {
 protected:
  Program *prog;
  Kernel *kernel;

  virtual FunctionType codegen() = 0;

 public:
  KernelCodeGen(Kernel *kernel) : prog(&kernel->program), kernel(kernel) {
  }

  virtual FunctionType compile();

  static std::unique_ptr<KernelCodeGen> create(Arch arch, Kernel *kernel);

  virtual ~KernelCodeGen() = default;
};

TLANG_NAMESPACE_END