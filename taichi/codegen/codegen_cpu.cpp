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
  auto ir = kernel->ir;
  bool print_ir = false;
  if (kernel->is_accessor) {
    print_ir = prog->config.print_accessor_ir;
  } else {
    print_ir = prog->config.print_ir;
  }
  if (print_ir) {
    TI_TRACE("Initial IR:");
    irpass::re_id(ir);
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
  if (print_ir) {
    TI_TRACE("Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (print_ir) {
    TI_TRACE("Typechecked:");
    irpass::re_id(ir);
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
  irpass::slp_vectorize(ir);
  if (print_ir) {
    TI_TRACE("SLPed:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (print_ir) {
    TI_TRACE("LoopVeced:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (print_ir) {
    TI_TRACE("LoopSplitted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::simplify(ir);
  if (print_ir) {
    TI_TRACE("Simplified I:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (kernel->grad) {
    // irpass::re_id(ir);
    // TI_TRACE("Primal:");
    // irpass::print(ir);
    irpass::demote_atomics(ir);
    irpass::simplify(ir);
    irpass::make_adjoint(ir, true);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Adjoint:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (prog->config.debug) {
    irpass::check_out_of_bound(ir);
    if (print_ir) {
      TI_TRACE("Bound checked:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower_access(ir, true);
  if (print_ir) {
    TI_TRACE("Access Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
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

  irpass::constant_fold(ir);
  if (print_ir) {
    TI_TRACE("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::offload(ir);
  if (print_ir) {
    TI_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir, prog->config);
  if (print_ir) {
    TI_TRACE("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::demote_atomics(ir);
  if (print_ir) {
    TI_TRACE("Atomics demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  {
    // debugging here
    auto block = dynamic_cast<Block *>(ir);
    auto offload = block->statements[0].get()->cast<OffloadedStmt>();
    offload->body = std::make_unique<Block>();
    auto *body = offload->body.get();
    VecStatement new_body;
    auto alloc = new_body.push_back<StackAllocaStmt>(DataType::i32, 10);
    auto one = new_body.push_back<ConstStmt>(TypedConstant(42));
    new_body.push_back<StackPushStmt>(alloc, one);
    body->insert(std::move(new_body), 0);
    irpass::typecheck(ir);
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

TLANG_NAMESPACE_END
