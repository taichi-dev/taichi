// Driver class for kernel codegen

#include "codegen.h"
#include "codegen_cpu.h"
#include "codegen_cuda.h"
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile() {
  lower();
  return codegen();
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(Arch arch, Kernel *kernel) {
  if (arch_is_cpu(arch)) {
    return std::make_unique<CodeGenCPU>(kernel);
  } else if (arch == Arch::cuda) {
    return std::make_unique<CodeGenCUDA>(kernel);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
