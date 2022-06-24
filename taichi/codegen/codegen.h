// Driver class for kernel code generators.

#pragma once
#include "taichi/ir/ir.h"

#include "taichi/program/program.h"
#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#include "taichi/codegen/codegen_llvm.h"
#include "taichi/llvm/launch_arg_info.h"
#include "taichi/llvm/llvm_codegen_utils.h"
#endif

TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM
class ModuleGenValue {
 public:
  ModuleGenValue(std::unique_ptr<llvm::Module> module,
                 const std::vector<std::string> &name_list)
      : module(std::move(module)), name_list(name_list) {
  }
  std::unique_ptr<llvm::Module> module;
  std::vector<std::string> name_list;
};
#endif

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

  virtual FunctionType codegen() = 0;
#ifdef TI_WITH_LLVM
  virtual std::unique_ptr<ModuleGenValue> modulegen(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) {
    TI_NOT_IMPLEMENTED
  }
#endif
};

class ModuleToFunctionConverter {
 public:
  explicit ModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                     LlvmProgramImpl *program);

  virtual ~ModuleToFunctionConverter() = default;

  virtual FunctionType convert(const std::string &kernel_name,
                               const std::vector<LlvmLaunchArgInfo> &args,
                               std::unique_ptr<llvm::Module> mod,
                               std::vector<OffloadedTask> &&tasks) const = 0;

  virtual FunctionType convert(const Kernel *kernel,
                               std::unique_ptr<llvm::Module> mod,
                               std::vector<OffloadedTask> &&tasks) const;

 protected:
  TaichiLLVMContext *tlctx_{nullptr};
  LlvmProgramImpl *program_{nullptr};
};

TLANG_NAMESPACE_END
