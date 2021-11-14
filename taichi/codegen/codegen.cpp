// Driver class for kernel codegen

#include "codegen.h"

#include "taichi/util/statistics.h"
#if defined(TI_WITH_LLVM)
#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/backends/wasm/codegen_wasm.h"
#endif
#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/codegen_cuda.h"
#endif
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

KernelCodeGen::KernelCodeGen(Kernel *kernel, IRNode *ir)
    : prog(kernel->program), kernel(kernel), ir(ir) {
  if (ir == nullptr)
    this->ir = kernel->ir.get();

  auto num_stmts = irpass::analysis::count_statements(this->ir);
  if (kernel->is_evaluator)
    stat.add("codegen_evaluator_statements", num_stmts);
  else if (kernel->is_accessor)
    stat.add("codegen_accessor_statements", num_stmts);
  else
    stat.add("codegen_kernel_statements", num_stmts);
  stat.add("codegen_statements", num_stmts);
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(Arch arch,
                                                     Kernel *kernel,
                                                     Stmt *stmt) {
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch) && arch != Arch::wasm) {
    return std::make_unique<CodeGenCPU>(kernel, stmt);
  } else if (arch == Arch::wasm) {
    return std::make_unique<CodeGenWASM>(kernel, stmt);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<CodeGenCUDA>(kernel, stmt);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}

TLANG_NAMESPACE_END
