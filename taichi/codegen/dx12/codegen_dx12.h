// dx12 backend implementation

#pragma once

#include <memory>

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/aot/module_data.h"

TLANG_NAMESPACE_BEGIN

class KernelCodeGenDX12 : public KernelCodeGen {
 public:
  KernelCodeGenDX12(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }
  struct CompileResult {
    std::vector<std::vector<uint8_t>> task_dxil_source_codes;
    std::vector<aot::CompiledOffloadedTask> tasks;
    std::size_t num_snode_trees{0};
  };
  CompileResult compile();
#ifdef TI_WITH_LLVM
  LLVMCompiledData compile_task(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) override;
#endif
  FunctionType compile_to_function() override;
};

TLANG_NAMESPACE_END
