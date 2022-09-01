// x86 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGenCPU : public KernelCodeGen {
 public:
  KernelCodeGenCPU(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  // TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  static std::unique_ptr<TaskCodeGenLLVM> make_codegen_llvm(Kernel *kernel,
                                                            IRNode *ir);

  bool supports_offline_cache() const override {
    return true;
  }
  LLVMCompiledData compile_task(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) override;

#endif  // TI_WITH_LLVM

  FunctionType compile_to_function() override;
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
