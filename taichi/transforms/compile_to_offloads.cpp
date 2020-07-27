#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
namespace {

std::function<void(const std::string &)> make_pass_printer(bool verbose,
                                                           IRNode *ir) {
  if (!verbose) {
    return [](const std::string &) {};
  }
  return [ir, kn = ir->get_kernel()->name](const std::string &pass) {
    TI_INFO("[{}] {}:", kn, pass);
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  };
}

}  // namespace

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         bool verbose,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, ir);
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
  irpass::full_simplify(ir, false);
  print("Simplified I");
  irpass::analysis::verify(ir);

  if (grad) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir);

    irpass::full_simplify(ir, false);
    irpass::auto_diff(ir, ad_use_stack);
    irpass::full_simplify(ir, false);
    print("Gradient");
    irpass::analysis::verify(ir);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir);
    print("Bound checked");
    irpass::analysis::verify(ir);
  }

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);

  irpass::full_simplify(ir, false);
  print("Simplified II");
  irpass::analysis::verify(ir);

  irpass::offload(ir);
  print("Offloaded");
  irpass::analysis::verify(ir);

  irpass::cfg_optimization(ir, false);
  print("Optimized by CFG");
  irpass::analysis::verify(ir);

  irpass::flag_access(ir);

  print("Access flagged II");
  irpass::analysis::verify(ir);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           bool vectorize,
                           bool grad,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, ir);

  compile_to_offloads(ir, config, verbose, vectorize, grad, ad_use_stack);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.
  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir);
    print("Make thread local");
  }

  if (make_block_local) {
    irpass::make_block_local(ir);
    print("Make block local");
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

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

  irpass::full_simplify(ir, lower_global_access);
  print("Simplified III");

  // Final field registration correctness & type checking
  irpass::typecheck(ir);
  irpass::analysis::verify(ir);
}

}  // namespace irpass

TLANG_NAMESPACE_END
