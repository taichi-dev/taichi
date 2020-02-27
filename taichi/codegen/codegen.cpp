// Driver class for kernel codegen

#include "codegen.h"
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile() {
  lower();
  return codegen();
}

TLANG_NAMESPACE_END
