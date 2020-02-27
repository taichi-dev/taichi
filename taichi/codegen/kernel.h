// Driver class for source2source kernel codegen. This file is OBSOLETE

#pragma once

#include "taichi/tlang_util.h"
#include "taichi/program.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGen {
 public:
  Program *prog;
  Kernel *kernel;

  KernelCodeGen(const std::string &kernel_name) {
  }

  virtual void lower() = 0;

  virtual FunctionType codegen() = 0;

  virtual FunctionType compile(Program &prog, Kernel &kernel);
};

TLANG_NAMESPACE_END