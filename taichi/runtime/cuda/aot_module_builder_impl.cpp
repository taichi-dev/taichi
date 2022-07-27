#include "taichi/runtime/cuda/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/cuda/codegen_cuda.h"
#include "taichi/runtime/llvm/launch_arg_info.h"

namespace taichi {
namespace lang {
namespace cuda {

LLVMCompiledData AotModuleBuilderImpl::compile_kernel(Kernel *kernel) {
  auto cgen = KernelCodeGenCUDA::make_codegen_llvm(kernel, /*ir=*/nullptr);
  return cgen->run_compilation();
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
