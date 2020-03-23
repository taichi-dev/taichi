// The CUDA backend

#include "codegen_cuda.h"
#include "taichi/ir/scratch_pad.h"

TLANG_NAMESPACE_BEGIN

void CodeGenCUDA::lower() {
  auto ir = kernel->ir;
  bool print_ir = false;
  if (kernel->is_accessor) {
    print_ir = prog->config.print_accessor_ir;
  } else {
    print_ir = prog->config.print_ir;
  }
  if (print_ir) {
    irpass::print(ir);
  }
  if (kernel->grad) {
    irpass::reverse_segments(ir);
    if (print_ir) {
      TI_TRACE("Segment reversed (for autodiff):");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower(ir);
  irpass::re_id(ir);
  if (print_ir) {
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  irpass::re_id(ir);
  if (print_ir) {
    irpass::print(ir);
  }
  if (!kernel->grad && prog->config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Dense Struct-for demoted:");
      irpass::print(ir);
    }
  }
  irpass::constant_fold(ir);
  if (prog->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TI_TRACE("Simplified I:");
      irpass::print(ir);
    }
  }
  if (kernel->grad) {
    // irpass::re_id(ir);
    // TI_TRACE("Primal:");
    // irpass::print(ir);
    irpass::demote_atomics(ir);
    irpass::full_simplify(ir, prog->config);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Before make_adjoint:");
      irpass::print(ir);
    }
    irpass::make_adjoint(ir, true);
    if (print_ir) {
      TI_TRACE("After make_adjoint:");
      irpass::print(ir);
    }
    irpass::typecheck(ir);
    // irpass::re_id(ir);
    // TI_TRACE("Adjoint:");
    // irpass::print(ir);
  }
  irpass::lower_access(ir, prog->config.use_llvm);
  if (print_ir) {
    TI_TRACE("Access Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (prog->config.simplify_after_lower_access) {
    irpass::die(ir);
    if (print_ir) {
      TI_TRACE("DIEd:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    irpass::simplify(ir);
    if (print_ir) {
      TI_TRACE("Simplified II:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::die(ir);
  if (print_ir) {
    TI_TRACE("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::flag_access(ir);
  if (print_ir) {
    TI_TRACE("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::offload(ir);
  if (print_ir) {
    TI_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::demote_atomics(ir);
  if (print_ir) {
    TI_TRACE("Atomics Demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::full_simplify(ir, prog->config);
  if (print_ir) {
    TI_TRACE("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

TLANG_NAMESPACE_END
