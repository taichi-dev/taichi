#include "taichi/backends/cpu/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/backends/cpu/codegen_cpu.h"
#include "taichi/llvm/launch_arg_info.h"

namespace taichi {
namespace lang {
namespace cpu {

CodeGenLLVM::CompiledData AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = CodeGenCPU::make_codegen_llvm(kernel, /*ir=*/nullptr);
  return cgen->run_compilation();
}

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
