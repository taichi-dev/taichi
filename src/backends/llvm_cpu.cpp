// A work-in-progress llvm backend

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../util.h"
#include "codegen_x86.h"
#include "../program.h"
#include "../ir.h"

#if defined(TLANG_WITH_LLVM)
#include "codegen_llvm.h"
#endif

TLANG_NAMESPACE_BEGIN

#if defined(TLANG_WITH_LLVM)

using namespace llvm;
class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  CodeGenLLVMCPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }
};

FunctionType CPUCodeGen::codegen_llvm() {
  TC_PROFILER("cpu codegen");
  return CodeGenLLVMCPU(this, kernel).gen();
}
#else

FunctionType CPUCodeGen::codegen_llvm() {
  TC_ERROR("LLVM not found");
}

#endif

TLANG_NAMESPACE_END
