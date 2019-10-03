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

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf
int compile_ptx_and_launch2() {
  printf("123321\n");
  // return 1;
}

class CodeGenLLVMGPU : public CodeGenLLVM {
 public:
  CodeGenLLVMGPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
  }

  FunctionType compile_module_to_executable() override {
    // llvm::Function *func = module->getFunction("test_kernel");

    /*******************************************************************
    Example annotation from llvm PTX doc:

    define void @kernel(float addrspace(1)* %A,
                        float addrspace(1)* %B,
                        float addrspace(1)* %C);

    !nvvm.annotations = !{!0}
    !0 = !{void (float addrspace(1)*,
                 float addrspace(1)*,
                 float addrspace(1)*)* @kernel, !"kernel", i32 1}
    *******************************************************************/

    // Mark kernel function as a CUDA __global__ function
    // Add the nvvm annotation that it is considered a kernel function.

    /*
    llvm::Metadata *md_args[] = {
        llvm::ValueAsMetadata::get(func),
        MDString::get(*llvm_context, "kernel"),
        llvm::ValueAsMetadata::get(tlctx->get_constant(1))};

    MDNode *md_node = MDNode::get(*llvm_context, md_args);
    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);
    */

    // auto ptx = compile_module_to_ptx(module);
    // TC_TAG;
    printf("123\n");
    compile_ptx_and_launch2();
    printf("456\n");
    // TC_TAG;
    // TC_NOT_IMPLEMENTED
    return nullptr;
  }

  void visit(PrintStmt *stmt) override {
    TC_ASSERT(stmt->width() == 1);

    auto value_type = tlctx->get_data_type(stmt->stmt->ret_type.data_type);
    std::vector<llvm::Type *> types{value_type};
    auto stype = llvm::StructType::get(*llvm_context, types, false);

    std::string format;

    auto values = builder->CreateAlloca(stype);
    auto value_ptr = builder->CreateGEP(
        values, {tlctx->get_constant(0), tlctx->get_constant(0)});
    auto value = stmt->stmt->value;

    if (stmt->stmt->ret_type.data_type == DataType::i32) {
      format = "%d";
    } else if (stmt->stmt->ret_type.data_type == DataType::f32) {
      format = "%f";
    } else {
      TC_NOT_IMPLEMENTED
    }

    builder->CreateStore(value, value_ptr);

    auto format_str = "[debug] " + stmt->str + " = " + format + "\n";

    stmt->value = ModuleBuilder::call(
        builder, "vprintf",
        builder->CreateGlobalStringPtr(format_str, "format_string"),
        builder->CreateBitCast(values,
                               llvm::Type::getInt8PtrTy(*llvm_context)));
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
