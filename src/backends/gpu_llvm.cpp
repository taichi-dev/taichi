// A work-in-progress llvm backend

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../util.h"
#include "gpu.h"
#include "../program.h"
#include "../ir.h"

#if defined(TLANG_WITH_LLVM)
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm_jit.h"

#endif

TLANG_NAMESPACE_BEGIN

#if defined(TLANG_WITH_LLVM)

using namespace llvm;

class GPULLVMCodeGen : public IRVisitor {
 public:
  GPULLVMCodeGen(CodeGenBase *codegen, Kernel *kernel) {
  }


  FunctionType gen(IRNode *node) {
    TC_NOT_IMPLEMENTED
  }

  static FunctionType run(CodeGenBase *codegen, IRNode *node, Kernel *kernel) {
    auto p = GPULLVMCodeGen(codegen, kernel);
    return p.gen(node);
  }
};

FunctionType GPUCodeGen::codegen_llvm() {
  return GPULLVMCodeGen::run(this, kernel->ir, kernel);
}
#else

FunctionType GPUCodeGen::codegen_llvm() {
  TC_ERROR("LLVM not found");
}

#endif

TLANG_NAMESPACE_END
