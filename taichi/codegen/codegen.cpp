// Driver class for kernel codegen

#include "codegen.h"

#include "taichi/util/statistics.h"
#include "taichi/backends/cpu/codegen_cpu.h"
#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/codegen_cuda.h"
#endif
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

KernelCodeGen::KernelCodeGen(Kernel *kernel, IRNode *ir)
    : prog(&kernel->program), kernel(kernel), ir(ir) {
  if (ir == nullptr)
    this->ir = kernel->ir;

  auto num_stmts = irpass::analysis::count_statements(this->ir);
  if (kernel->is_evaluator)
    stat.add("codegen_evaluator_statements", num_stmts);
  else if (kernel->is_accessor)
    stat.add("codegen_accessor_statements", num_stmts);
  else
    stat.add("codegen_kernel_statements", num_stmts);
  stat.add("codegen_statements", num_stmts);
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
#if defined(TI_WITH_CUDA)
    return std::make_unique<CodeGenCUDA>(kernel);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
