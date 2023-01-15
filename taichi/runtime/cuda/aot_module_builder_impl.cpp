#include "taichi/runtime/cuda/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cuda/codegen_cuda.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi::lang {
namespace cuda {

LLVMCompiledKernel AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = KernelCodeGenCUDA(get_compile_config(), kernel);
  return cgen.compile_kernel_to_module();
}

}  // namespace cuda
}  // namespace taichi::lang
