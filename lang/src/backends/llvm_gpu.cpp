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

// NVVM IR Spec: https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

class CodeGenLLVMGPU : public CodeGenLLVM {
 public:
  CodeGenLLVMGPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }

  FunctionType compile_module_to_executable() override {
    // Mark kernel function as a CUDA __global__ function
    llvm::Function *func = module->getFunction("test_kernel");

    // Example annotation from llvm PTX doc:
/*
define void @kernel(float addrspace(1)* %A,
                    float addrspace(1)* %B,
                    float addrspace(1)* %C);

!nvvm.annotations = !{!0}
!0 = !{void (float addrspace(1)*,
             float addrspace(1)*,
             float addrspace(1)*)* @kernel, !"kernel", i32 1}
*/

    // Add the nvvm annotation that it is a kernel function.
    llvm::Metadata *md_args[] = {
        llvm::ValueAsMetadata::get(func),
        MDString::get(*llvm_context, "kernel"),
        llvm::ValueAsMetadata::get(tlctx->get_constant(1))};

    MDNode *md_node = MDNode::get(*llvm_context, md_args);

    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);

    auto ptx = compile_module_to_ptx(module);
    TC_P(ptx);
    compile_ptx_and_launch(ptx, "test_kernel");
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
