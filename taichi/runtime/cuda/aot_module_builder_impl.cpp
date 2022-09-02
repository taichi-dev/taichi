#include "taichi/runtime/cuda/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cuda/codegen_cuda.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi {
namespace lang {
namespace cuda {

LLVMCompiledData AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = KernelCodeGenCUDA(kernel);
  return std::move(cgen.compile_kernel_to_module()[0]);
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
