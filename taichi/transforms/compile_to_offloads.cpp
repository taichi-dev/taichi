#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         CompileConfig config,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack,
                         bool verbose) {
  TI_AUTO_PROF;
  if (verbose) {
    TI_INFO("Initial IR:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (grad) {
    irpass::reverse_segments(ir);
    if (verbose) {
      TI_INFO("Segment reversed (for autodiff):");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower(ir);
  if (verbose) {
    TI_INFO("Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (verbose) {
    TI_INFO("Typechecked:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (vectorize) {
    irpass::slp_vectorize(ir);
    if (verbose) {
      TI_INFO("SLPed:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    irpass::loop_vectorize(ir);
    if (verbose) {
      TI_INFO("LoopVeced:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    irpass::vector_split(ir, config.max_vector_width, config.serial_schedule);
    if (verbose) {
      TI_INFO("LoopSplitted:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::simplify(ir);
  if (verbose) {
    TI_INFO("Simplified I:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (grad) {
    irpass::demote_atomics(ir);
    irpass::simplify(ir);
    irpass::make_adjoint(ir, ad_use_stack);
    irpass::full_simplify(ir, config);
    if (verbose) {
      TI_INFO("Adjoint:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    if (verbose) {
      TI_INFO("Dense Struct-for demoted:");
      irpass::print(ir);
    }
  }
  if (config.debug) {
    irpass::check_out_of_bound(ir);
    if (verbose) {
      TI_INFO("Bound checked:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower_access(ir, true);
  if (verbose) {
    TI_INFO("Access Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::die(ir);
  if (verbose) {
    TI_INFO("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::simplify(ir);
  if (verbose) {
    TI_INFO("Simplified II:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::die(ir);
  if (verbose) {
    TI_INFO("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  if (verbose) {
    TI_INFO("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (verbose) {
    TI_INFO("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::offload(ir);
  if (verbose) {
    TI_INFO("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir, config);
  if (verbose) {
    TI_INFO("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::demote_atomics(ir);
  if (verbose) {
    TI_INFO("Atomics demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
