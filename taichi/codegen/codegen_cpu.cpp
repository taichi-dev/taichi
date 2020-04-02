// x86 backend implementation

#include "taichi/common/util.h"
#include "taichi/util/io.h"
#include <set>
#include "codegen_cpu.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

void CodeGenCPU::lower() {
  auto config = kernel->program.config;
  bool verbose = config.print_ir;
  if (kernel->is_accessor && !config.print_accessor_ir) {
    verbose = false;
  }
  irpass::compile_to_offloads(kernel->ir, config, true, kernel->grad,
      /*ad_use_stack=*/true, verbose);
}

TLANG_NAMESPACE_END
