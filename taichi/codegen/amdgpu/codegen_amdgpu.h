// The AMDGPU backend
#pragma once

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi {
namespace lang {

class KernelCodeGenAMDGPU : public KernelCodeGen {
 public:
  KernelCodeGenAMDGPU(const CompileConfig &config,
                      const Kernel *kernel,
                      IRNode *ir,
                      TaichiLLVMContext &tlctx)
      : KernelCodeGen(config, kernel, ir, tlctx) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
#endif  // TI_WITH_LLVM
};

}  // namespace lang
}  // namespace taichi
