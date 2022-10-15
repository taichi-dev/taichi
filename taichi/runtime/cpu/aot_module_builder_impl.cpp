#include "taichi/runtime/cpu/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi::lang {
namespace cpu {

LLVMCompiledKernel AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = KernelCodeGenCPU(kernel);
  return cgen.compile_kernel_to_module();
}

}  // namespace cpu
}  // namespace taichi::lang
