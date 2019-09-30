// A work-in-progress llvm backend

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../util.h"
#include "gpu.h"
#include "../program.h"
#include "../ir.h"

#if defined(TLANG_WITH_LLVM)
#include "codegen_llvm.h"
#endif

TLANG_NAMESPACE_BEGIN

#if defined(TLANG_WITH_LLVM)

using namespace llvm;

class CodeGenLLVMGPU : public CodeGenLLVM {
 public:
  CodeGenLLVMGPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }

  FunctionType compile_module_to_executable() override {
    auto ptx = compile_module_to_ptx(module);
    TC_P(ptx);
    // return [=](Context context) { f(&context); };
    TC_NOT_IMPLEMENTED
    return nullptr;
  }
};

FunctionType GPUCodeGen::codegen_llvm() {
  return CodeGenLLVMGPU(this, kernel).gen();
}
#else

FunctionType GPUCodeGen::codegen_llvm() {
  TC_ERROR("LLVM not found");
}

#endif

TLANG_NAMESPACE_END
