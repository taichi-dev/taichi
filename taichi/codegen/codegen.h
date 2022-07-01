// Driver class for kernel code generators.

#pragma once
#include "taichi/ir/ir.h"

#include "taichi/program/program.h"
#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/runtime/llvm/launch_arg_info.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"
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
  virtual bool supports_offline_cache() const {
    return false;
  }

#ifdef TI_WITH_LLVM
  virtual LLVMCompiledData modulegen(
      std::unique_ptr<llvm::Module> &&module = nullptr,
      OffloadedStmt *stmt = nullptr) {
    TI_NOT_IMPLEMENTED
  }
  bool maybe_read_compilation_from_cache(
      const std::string &kernel_key,
                                         std::vector<LLVMCompiledData> &data);
#endif
};

class ModuleToFunctionConverter {
 public:
  explicit ModuleToFunctionConverter(TaichiLLVMContext *tlctx,
                                     LlvmProgramImpl *program);

  virtual ~ModuleToFunctionConverter() = default;

  virtual FunctionType convert(const std::string &kernel_name,
                               const std::vector<LlvmLaunchArgInfo> &args,
                               std::vector<LLVMCompiledData> &&data) const = 0;

  virtual FunctionType convert(const Kernel *kernel,
                               std::vector<LLVMCompiledData> &&data) const;

 protected:
  TaichiLLVMContext *tlctx_{nullptr};
  LlvmProgramImpl *program_{nullptr};
};

TLANG_NAMESPACE_END
