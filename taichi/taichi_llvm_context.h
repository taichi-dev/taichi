#pragma once
// A helper for the llvm backend

#include <functional>

#include "tlang_util.h"
#include "llvm_fwd.h"
#include "snode.h"

TLANG_NAMESPACE_BEGIN
class TaichiLLVMJITCPU;

void *jit_lookup_name(TaichiLLVMJITCPU *jit, const std::string &name);

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJITCPU> jit;
  std::unique_ptr<llvm::Module> runtime_module, struct_module;
  Arch arch;

  SNodeAttributes snode_attr;

  TaichiLLVMContext(Arch arch);

  std::unique_ptr<llvm::Module> get_init_module();

  std::unique_ptr<llvm::Module> clone_struct_module();

  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  virtual void *lookup_function_pointer(const std::string &name) {
    auto func_ptr = jit_lookup_name(jit.get(), name);
    return func_ptr;
  }

  // Unfortunately, this can't be virtual since it's a template function
  template <typename T>
  std::function<T> lookup_function(const std::string &name) {
    using FuncT = typename std::function<T>;
    auto ret =
        FuncT((function_pointer_type<FuncT>)lookup_function_pointer(name));
    TI_ASSERT(ret != nullptr);
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

  static void force_inline(llvm::Function *func);

  void print_huge_functions();

  static int num_instructions(llvm::Function *func);

  virtual ~TaichiLLVMContext();
};

TLANG_NAMESPACE_END
