#include "taichi/ir/transforms.h"
#include "taichi/program/extension.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace irpass {

static bool supports_lowering(Arch arch) {
  return arch_is_cpu(arch) || (arch == Arch::cuda) || (arch == Arch::dx12) ||
         (arch == Arch::metal) || (arch == Arch::amdgpu);
}

void ast_to_ir(const CompileConfig &config,
               Kernel &kernel,
               bool to_executable) {
  TI_ASSERT(supports_lowering(config.arch));
  if (kernel.lowered()) {
    return;
  }

  bool verbose = config.print_ir;
  if ((kernel.is_accessor && !config.print_accessor_ir) ||
      (kernel.is_evaluator && !config.print_evaluator_ir))
    verbose = false;

  if (to_executable) {
    irpass::compile_to_executable(
        kernel.ir.get(), config, &kernel,
        /*autodiff_mode=*/kernel.autodiff_mode,
        /*ad_use_stack=*/true,
        /*verbose*/ verbose,
        /*lower_global_access=*/to_executable,
        /*make_thread_local=*/config.make_thread_local,
        /*make_block_local=*/
        is_extension_supported(config.arch, Extension::bls) &&
            config.make_block_local,
        /*start_from_ast=*/kernel.ir_is_ast());
  } else {
    irpass::compile_to_offloads(kernel.ir.get(), config, &kernel, verbose,
                                /*autodiff_mode=*/kernel.autodiff_mode,
                                /*ad_use_stack=*/true,
                                /*start_from_ast=*/kernel.ir_is_ast());
  }

  kernel.set_lowered(true);
}

}  // namespace irpass
}  // namespace taichi::lang
