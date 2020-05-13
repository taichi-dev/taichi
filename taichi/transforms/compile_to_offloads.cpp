#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

OffloadedResult compile_to_offloads(
                         IRNode *ir,
                         const CompileConfig &config,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack,
                         bool verbose,
                         bool lower_global_access) {
  TI_AUTO_PROF;

  auto print = [&](const std::string &name) {
    if (verbose) {
      TI_INFO(name + ":");
      irpass::re_id(ir);
      irpass::print(ir);
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

  if (vectorize) {
    irpass::slp_vectorize(ir);
    print("SLP");

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

  if (grad) {
    irpass::demote_atomics(ir);
    irpass::full_simplify(ir, config);
    irpass::make_adjoint(ir, ad_use_stack);
    irpass::full_simplify(ir, config);
    print("Adjoint");
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

  irpass::extract_constant(ir);
  print("Constant extracted");
  irpass::analysis::verify(ir);

  irpass::variable_optimization(ir, false);
  print("Store forwarded");
  irpass::analysis::verify(ir);

  if (lower_global_access) {
    irpass::lower_access(ir, true);
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);
  }

  irpass::full_simplify(ir, config);
  print("Simplified II");
  irpass::analysis::verify(ir);

  irpass::flag_access(ir);
  print("Access flagged");
  irpass::analysis::verify(ir);

  irpass::constant_fold(ir);
  print("Constant folded");

  auto res = irpass::offload(ir);
  print("Offloaded");
  irpass::analysis::verify(ir);

  if (!lower_global_access) {
    irpass::flag_access(ir);
    print("Access flagged after offloading");
    irpass::analysis::verify(ir);
  }

  irpass::extract_constant(ir);
  print("Constant extracted II");

  irpass::demote_atomics(ir);
  print("Atomics demoted");
  irpass::analysis::verify(ir);

  irpass::variable_optimization(ir, true);
  print("Store forwarded II");

  irpass::full_simplify(ir, config);
  print("Simplified III");

  // Final field registration correctness & type checking
  irpass::typecheck(ir);
  irpass::analysis::verify(ir);

  return res;
}

}  // namespace irpass

TLANG_NAMESPACE_END
