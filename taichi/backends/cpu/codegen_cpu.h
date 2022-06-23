// x86 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/codegen_llvm.h"

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

class CPUModuleToFunctionConverter : public ModuleToFunctionConverter {
 public:
  explicit CPUModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                         LlvmProgramImpl *program)
      : ModuleToFunctionConverter(tlctx, program) {
  }

  using ModuleToFunctionConverter::convert;

  FunctionType convert(const std::string &kernel_name,
                       const std::vector<LlvmLaunchArgInfo> &args,
                       std::unique_ptr<llvm::Module> mod,
                       std::vector<OffloadedTask> &&tasks) const override;
};

TLANG_NAMESPACE_END
