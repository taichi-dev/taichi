// Driver class for kernel codegen

#include "codegen.h"

#include "taichi/util/statistics.h"
#include "taichi/codegen/codegen_cpu.h"
#include "taichi/codegen/codegen_cuda.h"
#include "taichi/system/timer.h"
#include "taichi/system/timer.h"

TLANG_NAMESPACE_BEGIN

KernelCodeGen::KernelCodeGen(Kernel *kernel, IRNode *ir)
    : prog(&kernel->program), kernel(kernel), ir(ir) {
  if (ir == nullptr)
    this->ir = kernel->ir;

  stat.add("codegen_statements", irpass::analysis::count_statements(this->ir));
}

FunctionType KernelCodeGen::compile() {
  TI_AUTO_PROF;
  return codegen();
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(Arch arch,
                                                     Kernel *kernel) {
  if (arch_is_cpu(arch)) {
    return std::make_unique<CodeGenCPU>(kernel);
  } else if (arch == Arch::cuda) {
    return std::make_unique<CodeGenCUDA>(kernel);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
