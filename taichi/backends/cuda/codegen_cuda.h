// The CUDA backend

#pragma once

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCUDA : public KernelCodeGen {
 public:
  CodeGenCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  static std::unique_ptr<CodeGenLLVM> make_codegen_llvm(Kernel *kernel,
                                                        IRNode *ir);
#endif  // TI_WITH_LLVM

  FunctionType codegen() override;
};

class CUDAModuleToFunctionConverter : public ModuleToFunctionConverter {
 public:
  explicit CUDAModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                         LlvmProgramImpl *program)
      : ModuleToFunctionConverter(tlctx, program) {
  }

  FunctionType convert(const std::string &kernel_name,
                       const std::vector<LlvmLaunchArgInfo> &args,
                       std::unique_ptr<llvm::Module> mod,
                       std::vector<OffloadedTask> &&tasks) const override;

  FunctionType convert(const Kernel *kernel,
                       std::unique_ptr<llvm::Module> mod,
                       std::vector<OffloadedTask> &&tasks) const override;
};

TLANG_NAMESPACE_END
