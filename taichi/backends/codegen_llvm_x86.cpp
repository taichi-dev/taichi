#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include "../tlang_util.h"
#include "../program.h"
#include "../ir.h"
#include "codegen_x86.h"
#include "codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  CodeGenLLVMCPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }
};

FunctionType CPUCodeGen::codegen_llvm() {
  TI_PROFILER("cpu codegen");
  return CodeGenLLVMCPU(this, kernel).gen();
}

TLANG_NAMESPACE_END
