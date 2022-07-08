// x86 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCPU : public KernelCodeGen {
public:
  CodeGenCPU(Kernel *kernel, IRNode *ir = nullptr) : KernelCodeGen(kernel, ir) {
  }

  // TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  static std::unique_ptr<CodeGenLLVM> make_codegen_llvm(Kernel *kernel,
                                                        IRNode *ir);
#endif  // TI_WITH_LLVM

  FunctionType codegen() override;
};

#ifdef TI_WITH_LLVM

class CPUModuleToFunctionConverter : public ModuleToFunctionConverter {
 public:
  explicit CPUModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                        LlvmRuntimeExecutor *executor)
      : ModuleToFunctionConverter(tlctx, executor) {
  }

  using ModuleToFunctionConverter::convert;

  FunctionType convert(const std::string &kernel_name,
                       const std::vector<LlvmLaunchArgInfo> &args,
                       std::vector<LLVMCompiledData> &&data) const override;
};

#endif

TLANG_NAMESPACE_END
