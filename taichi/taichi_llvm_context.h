#pragma once
// A helper for the llvm backend

#include "tlang_util.h"
#include "llvm_fwd.h"
#include "snode.h"

TLANG_NAMESPACE_BEGIN
class TaichiLLVMJIT;

void *jit_lookup_name(TaichiLLVMJIT *jit, const std::string &name);

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJIT> jit;
  std::unique_ptr<llvm::Module> runtime_module, struct_module;
  Arch arch;

  SNodeAttributes snode_attr;

  TaichiLLVMContext(Arch arch);

  ~TaichiLLVMContext();

  std::unique_ptr<llvm::Module> get_init_module();

  std::unique_ptr<llvm::Module> clone_struct_module();

  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  template <typename T>
  T lookup_function(const std::string &name) {
    auto ret = T((function_pointer_type<T>)jit_lookup_name(jit.get(), name));
    TC_ASSERT(ret != nullptr);
    return ret;
  }

  std::unique_ptr<llvm::Module> clone_runtime_module();

  llvm::Type *get_data_type(DataType dt);

  template <typename T>
  llvm::Type *get_data_type() {
    return TaichiLLVMContext::get_data_type(taichi::Tlang::get_data_type<T>());
  }

  std::size_t get_type_size(llvm::Type *type);

  template <typename T>
  llvm::Value *get_constant(T t);

  template <typename T>
  llvm::Value *get_constant(DataType dt, T t);

  std::string type_name(llvm::Type *type);

  void link_module_with_libdevice(std::unique_ptr<llvm::Module> &module);
};

TLANG_NAMESPACE_END
