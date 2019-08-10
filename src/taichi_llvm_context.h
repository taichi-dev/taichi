#pragma once
#include "util.h"
#include "llvm_fwd.h"

TLANG_NAMESPACE_BEGIN

class TaichiLLVMJIT;

void *jit_lookup_name(TaichiLLVMJIT *jit, const std::string &name);

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJIT> jit;
  std::unique_ptr<llvm::Module> struct_module, runtime_module;

  llvm::Type *get_data_type(DataType dt);

  TaichiLLVMContext();

  ~TaichiLLVMContext();

  std::unique_ptr<llvm::Module> clone_struct_module();

  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  template <typename T>
  llvm::Value *get_constant(T t);

  std::string type_name(llvm::Type *type);

  template <typename T>
  T lookup_function(const std::string &name) {
    return T((function_pointer_type<T>)jit_lookup_name(jit.get(), name));
  }

  std::unique_ptr<llvm::Module> clone_runtime_module();

  std::size_t get_type_size(llvm::Type *type);
};

TLANG_NAMESPACE_END
