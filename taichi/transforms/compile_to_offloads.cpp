#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack,
                         bool verbose,
                         bool lower_global_access,
                         bool make_thread_local) {
  TI_AUTO_PROF;

  auto print = [&](const std::string &name) {
    if (verbose) {
      TI_INFO(name + ":");
      std::cout << std::flush;
      irpass::re_id(ir);
      irpass::print(ir);
      std::cout << std::flush;
    }
  };

  print("Initial IR");

  if (grad) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  irpass::lower(ir);
  print("Lowered");

  irpass::typecheck(ir);
  print("Typechecked");
  irpass::analysis::verify(ir);

  if (ir->get_kernel()->is_evaluator) {
    TI_ASSERT(!grad);
    irpass::offload(ir);
    print("Offloaded");
    irpass::analysis::verify(ir);
    return;
  }

  if (vectorize) {
    irpass::loop_vectorize(ir);
    print("Loop Vectorized");
    irpass::analysis::verify(ir);

    irpass::vector_split(ir, config.max_vector_width, config.serial_schedule);
    print("Loop Split");
    irpass::analysis::verify(ir);
  }
  irpass::simplify(ir);
  print("Simplified I");
  irpass::analysis::verify(ir);

  irpass::constant_fold(ir);
  print("Constant folded I");

  if (grad) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir);

    irpass::full_simplify(ir);
    irpass::auto_diff(ir, ad_use_stack);
    irpass::full_simplify(ir);
    print("Gradient");
    // TODO: removing the following line break the verify pass. Need to figure
    // out why.
    irpass::fix_block_parents(ir);
    irpass::analysis::verify(ir);
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir);
    print("Bound checked");
    irpass::analysis::verify(ir);
  }

  irpass::cfg_optimization(ir, false);
  print("Optimized by CFG I");
  irpass::analysis::verify(ir);

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);

  irpass::full_simplify(ir);
  print("Simplified II");
  irpass::analysis::verify(ir);

  irpass::constant_fold(ir);
  print("Constant folded II");

  irpass::offload(ir);
  print("Offloaded");
  irpass::analysis::verify(ir);

  irpass::cfg_optimization(ir, false);
  print("Optimized by CFG II");
  irpass::analysis::verify(ir);

  irpass::flag_access(ir);
  print("Access flagged II");
  irpass::analysis::verify(ir);

  if (make_thread_local) {
    irpass::make_thread_local(ir);
    print("Make thread local");
  }

  if (lower_global_access) {
    irpass::lower_access(ir, true);
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify(ir);
  }

  irpass::demote_atomics(ir);
  print("Atomics demoted");
  irpass::analysis::verify(ir);

  irpass::cfg_optimization(ir, true);
  print("Optimized by CFG III");
  irpass::analysis::verify(ir);

  irpass::full_simplify(ir);
  print("Simplified III");

  // Final field registration correctness & type checking
  irpass::typecheck(ir);
  irpass::analysis::verify(ir);
}

}  // namespace irpass

TLANG_NAMESPACE_END
