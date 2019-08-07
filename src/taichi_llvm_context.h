#pragma once
#include "util.h"
#include "llvm_fwd.h"

TLANG_NAMESPACE_BEGIN

class TaichiLLVMJIT;

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJIT> jit;
  std::unique_ptr<llvm::Module> struct_module;

  llvm::Type *get_data_type(DataType dt);

  TaichiLLVMContext();

  ~TaichiLLVMContext();

  std::unique_ptr<llvm::Module> clone_struct_module();

  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  template <typename T>
  llvm::Value *get_constant(T t);

  std::string type_name(llvm::Type *type);
};

TLANG_NAMESPACE_END
