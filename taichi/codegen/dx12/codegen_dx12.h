// dx12 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/aot/module_data.h"

namespace taichi::lang {

class KernelCodeGenDX12 : public KernelCodeGen {
 public:
  explicit KernelCodeGenDX12(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             TaichiLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx) {
  }
  struct CompileResult {
    std::vector<std::vector<uint8_t>> task_dxil_source_codes;
    std::vector<aot::CompiledOffloadedTask> tasks;
    std::size_t num_snode_trees{0};
  };
  CompileResult compile();
#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
#endif
};

}  // namespace taichi::lang
