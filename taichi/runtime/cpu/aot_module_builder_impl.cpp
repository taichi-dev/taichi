#include "taichi/runtime/cpu/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi {
namespace lang {
namespace cpu {

LLVMCompiledData AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = KernelCodeGenCPU::make_codegen_llvm(kernel, /*ir=*/nullptr);
  return cgen->run_compilation();
}

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
