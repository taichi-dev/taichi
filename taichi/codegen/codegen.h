// Driver class for kernel code generators.

#pragma once
#include <taichi/runtime/llvm/llvm_runtime_executor.h>
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/runtime/llvm/launch_arg_info.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"
#endif
namespace taichi::lang {

class KernelCodeGen {
 protected:
  Program *prog;
  Kernel *kernel;
  IRNode *ir;

 public:
  KernelCodeGen(Kernel *kernel, IRNode *ir);

  virtual ~KernelCodeGen() = default;

  static std::unique_ptr<KernelCodeGen> create(Arch arch,
                                               Kernel *kernel,
                                               Stmt *stmt = nullptr);

  virtual FunctionType compile_to_function() = 0;
  virtual bool supports_offline_cache() const {
    return false;
  }

#ifdef TI_WITH_LLVM
  virtual LLVMCompiledKernel compile_kernel_to_module();

  virtual LLVMCompiledTask compile_task(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr){TI_NOT_IMPLEMENTED}

  std::optional<LLVMCompiledKernel> maybe_read_compilation_from_cache(
      const std::string &kernel_key);
  void cache_kernel(const std::string &kernel_key,
                    const LLVMCompiledKernel &data);
#endif
};

#ifdef TI_WITH_LLVM

class ModuleToFunctionConverter {
 public:
  explicit ModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                     LlvmRuntimeExecutor *program);

  virtual ~ModuleToFunctionConverter() = default;

  virtual FunctionType convert(const std::string &kernel_name,
                               const std::vector<LlvmLaunchArgInfo> &args,
                               LLVMCompiledKernel data) const = 0;

  virtual FunctionType convert(const Kernel *kernel,
                               LLVMCompiledKernel data) const;

 protected:
  TaichiLLVMContext *tlctx_{nullptr};
  LlvmRuntimeExecutor *executor_{nullptr};
};

#endif
}  // namespace taichi::lang
