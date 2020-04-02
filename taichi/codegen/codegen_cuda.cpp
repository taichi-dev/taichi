// The CUDA backend

#include "codegen_cuda.h"
#include "taichi/ir/scratch_pad.h"

TLANG_NAMESPACE_BEGIN

void CodeGenCUDA::lower() {
  auto config = kernel->program.config;
  bool verbose = config.print_ir;
  if (kernel->is_accessor && !config.print_accessor_ir) {
    verbose = false;
  }
  irpass::compile_to_offloads(kernel->ir, config, false, kernel->grad,
                              /*ad_use_stack=*/true, verbose);
}

TLANG_NAMESPACE_END
