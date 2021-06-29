#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
namespace {

std::function<void(const std::string &)>
make_pass_printer(bool verbose, const std::string &kernel_name, IRNode *ir) {
  if (!verbose) {
    return [](const std::string &) {};
  }
  return [ir, kernel_name](const std::string &pass) {
    TI_INFO("[{}] {}:", kernel_name, pass);
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  };
}

}  // namespace

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         Kernel *kernel,
                         bool verbose,
                         bool vectorize,
                         bool grad,
                         bool ad_use_stack,
                         bool start_from_ast) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, kernel->get_name(), ir);
  print("Initial IR");

  if (grad) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    irpass::lower_ast(ir);
    print("Lowered");
  }

  irpass::type_check(ir, config);
  print("Typechecked");
  irpass::analysis::verify(ir);

  if (kernel->is_evaluator) {
    TI_ASSERT(!grad);

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    irpass::offload(ir, config);
    print("Offloaded");
    irpass::analysis::verify(ir);
    return;
  }

  if (vectorize) {
    irpass::loop_vectorize(ir, config);
    print("Loop Vectorized");
    irpass::analysis::verify(ir);

    irpass::vector_split(ir, config.max_vector_width, config.serial_schedule);
    print("Loop Split");
    irpass::analysis::verify(ir);
  }

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    print("Bit Loop Vectorized");
    irpass::analysis::verify(ir);
  }

  irpass::full_simplify(ir, config, {false, kernel->program});
  print("Simplified I");
  irpass::analysis::verify(ir);

  if (irpass::inlining(ir, config, {})) {
    print("Functions inlined");
    irpass::analysis::verify(ir);
  }

  if (grad) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(ir, config, {false, kernel->program});
    irpass::auto_diff(ir, config, ad_use_stack);
    irpass::full_simplify(ir, config, {false, kernel->program});
    print("Gradient");
    irpass::analysis::verify(ir);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    print("Bound checked");
    irpass::analysis::verify(ir);
  }

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);

  irpass::full_simplify(ir, config, {false, kernel->program});
  print("Simplified II");
  irpass::analysis::verify(ir);

  irpass::offload(ir, config);
  print("Offloaded");
  irpass::analysis::verify(ir);

  // TODO: This pass may be redundant as cfg_optimization() is already called
  //  in full_simplify().
  if (config.cfg_optimization) {
    irpass::cfg_optimization(ir, false);
    print("Optimized by CFG");
    irpass::analysis::verify(ir);
  }

  irpass::flag_access(ir);
  print("Access flagged II");

  irpass::full_simplify(ir, config, {false, kernel->program});
  print("Simplified III");
  irpass::analysis::verify(ir);
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify(ir);

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted I");
  irpass::analysis::verify(ir);

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name()});
    print("Make block local");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted II");
  irpass::analysis::verify(ir);

  if (is_extension_supported(config.arch, Extension::quant) &&
      ir->get_config().quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify(ir);

  if (lower_global_access) {
    irpass::lower_access(ir, config, {kernel->no_activate, true});
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify(ir);
  }

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config, {lower_global_access, kernel->program});
  print("Simplified IV");

  if (determine_ad_stack_size) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify(ir);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           Kernel *kernel,
                           bool vectorize,
                           bool grad,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  TI_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, vectorize, grad,
                      ad_use_stack, start_from_ast);

  offload_to_executable(ir, config, kernel, verbose,
                        /*determine_ad_stack_size=*/grad && ad_use_stack,
                        lower_global_access, make_thread_local,
                        make_block_local);
}

void compile_inline_function(IRNode *ir,
                             const CompileConfig &config,
                             Function *func,
                             bool grad,
                             bool verbose,
                             bool start_from_ast) {
  TI_AUTO_PROF;

  auto print = make_pass_printer(verbose, func->get_name(), ir);
  print("Initial IR");

  if (grad) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    irpass::lower_ast(ir);
    print("Lowered");
  }

  irpass::type_check(ir, config);
  print("Typechecked");

  irpass::full_simplify(ir, config, {false, func->program});
  print("Simplified");
  irpass::analysis::verify(ir);
}

}  // namespace irpass

TLANG_NAMESPACE_END
