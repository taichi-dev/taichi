// x86 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class KernelCodeGenCPU : public KernelCodeGen {
 public:
  explicit KernelCodeGenCPU(const CompileConfig &compile_config,
                            const Kernel *kernel,
                            IRNode *ir,
                            TaichiLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx) {
  }

  // TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) override;

#endif  // TI_WITH_LLVM
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
                       const std::vector<Callable::Parameter> &args,
                       LLVMCompiledKernel data) const override;
};

#endif

}  // namespace taichi::lang
