#include "taichi/codegen/dx12/codegen_dx12.h"


#include "taichi/rhi/dx12/dx12_api.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"
TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM

KernelCodeGenDX12::CompileResult KernelCodeGenDX12::compile() {
  TI_NOT_IMPLEMENTED;
}

LLVMCompiledData KernelCodeGenDX12::modulegen(
    std::unique_ptr<llvm::Module> &&module,
    OffloadedStmt *stmt) {
  TI_NOT_IMPLEMENTED;
}
#endif  // TI_WITH_LLVM

FunctionType KernelCodeGenDX12::codegen() {
  // FIXME: implement codegen.
  return [](RuntimeContext &ctx) {
  };
}
TLANG_NAMESPACE_END
