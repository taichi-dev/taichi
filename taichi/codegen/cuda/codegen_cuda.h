// The CUDA backend

#pragma once

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGenCUDA : public KernelCodeGen {
 public:
  KernelCodeGenCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  static std::unique_ptr<TaskCodeGenLLVM> make_codegen_llvm(Kernel *kernel,
                                                            IRNode *ir);
  LLVMCompiledData modulegen(std::unique_ptr<llvm::Module> &&module = nullptr,
                             OffloadedStmt *stmt = nullptr) override;
#endif  // TI_WITH_LLVM

  bool supports_offline_cache() const override {
    return true;
  }

  FunctionType codegen() override;
};

class CUDAModuleToFunctionConverter : public ModuleToFunctionConverter {
 public:
  explicit CUDAModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                         LlvmRuntimeExecutor *executor)
      : ModuleToFunctionConverter(tlctx, executor) {
  }
  using ModuleToFunctionConverter::convert;

  FunctionType convert(const std::string &kernel_name,
                       const std::vector<LlvmLaunchArgInfo> &args,
                       std::vector<LLVMCompiledData> &&data) const override;
};

TLANG_NAMESPACE_END
