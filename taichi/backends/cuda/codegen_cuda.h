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
