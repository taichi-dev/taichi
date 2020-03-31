// x86 backend implementation

#include <taichi/common/util.h>
#include <taichi/util/io.h>
#include <set>
#include "codegen_cpu.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

void CodeGenCPU::lower() {
  TI_AUTO_PROF;
  auto ir = kernel->ir;
  bool print_ir = false;
  if (kernel->is_accessor) {
    print_ir = prog->config.print_accessor_ir;
  } else {
    print_ir = prog->config.print_ir;
  }
  if (print_ir) {
    TI_INFO("Initial IR:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (kernel->grad) {
    irpass::reverse_segments(ir);
    if (print_ir) {
      TI_INFO("Segment reversed (for autodiff):");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower(ir);
  if (print_ir) {
    TI_INFO("Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (print_ir) {
    TI_INFO("Typechecked:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (print_ir) {
    TI_INFO("SLPed:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (print_ir) {
    TI_INFO("LoopVeced:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (print_ir) {
    TI_INFO("LoopSplitted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::simplify(ir);
  if (print_ir) {
    TI_INFO("Simplified I:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (kernel->grad) {
    irpass::demote_atomics(ir);
    irpass::simplify(ir);
    irpass::make_adjoint(ir, true);
    irpass::full_simplify(ir, prog->config);
    if (print_ir) {
      TI_INFO("Adjoint:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (prog->config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_INFO("Dense Struct-for demoted:");
      irpass::print(ir);
    }
  }
  if (prog->config.debug) {
    irpass::check_out_of_bound(ir);
    if (print_ir) {
      TI_INFO("Bound checked:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower_access(ir, true);
  if (print_ir) {
    TI_INFO("Access Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::die(ir);
  if (print_ir) {
    TI_INFO("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::simplify(ir);
  if (print_ir) {
    TI_INFO("Simplified II:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::die(ir);
  if (print_ir) {
    TI_INFO("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  if (print_ir) {
    TI_INFO("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (print_ir) {
    TI_INFO("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::offload(ir);
  if (print_ir) {
    TI_INFO("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir, prog->config);
  if (print_ir) {
    TI_INFO("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::demote_atomics(ir);
  if (print_ir) {
    TI_INFO("Atomics demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

TLANG_NAMESPACE_END
