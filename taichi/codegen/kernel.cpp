// Driver class for kernel codegen

#include "kernel.h"
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile(taichi::Tlang::Program &prog,
                                    taichi::Tlang::Kernel &kernel) {
  this->prog = &kernel.program;
  this->kernel = &kernel;
  lower();
  return codegen();
}

TLANG_NAMESPACE_END
