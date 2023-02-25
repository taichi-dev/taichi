#include "taichi/runtime/cpu/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi::lang {
namespace cpu {

LLVMCompiledKernel AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen =
      KernelCodeGenCPU(get_compile_config(), kernel, get_taichi_llvm_context());
  return cgen.compile_kernel_to_module();
}

}  // namespace cpu
}  // namespace taichi::lang
