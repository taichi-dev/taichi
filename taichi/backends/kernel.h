// Driver class for source2source kernel codegen. This file is OBSOLETE

#pragma once

#include "base.h"
#include "../tlang_util.h"
#include "../program.h"

TLANG_NAMESPACE_BEGIN

class Program;
class KernelCodeGen : public CodeGenBase {
 public:
  Program *prog;
  Kernel *kernel;
  KernelCodeGen(const std::string &kernel_name) : CodeGenBase(kernel_name) {
  }

  virtual void generate_header() {
  }

  virtual void generate_tail() {
  }

  virtual void lower() = 0;

  virtual void codegen() {
    TI_NOT_IMPLEMENTED
  }

  virtual FunctionType codegen_llvm() {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  virtual FunctionType compile(Program &prog, Kernel &kernel);
};

TLANG_NAMESPACE_END