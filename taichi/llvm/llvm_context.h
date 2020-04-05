#pragma once

// llvm backend compiler (x64, arm64, cuda, amdgpu etc)
// in charge of creating & JITing arch-specific LLVM modules,
// and invoking compiled functions (kernels).

#include <functional>

#include "taichi/lang_util.h"
#include "llvm_fwd.h"
#include "taichi/ir/snode.h"
#include "taichi/jit/jit_session.h"

TLANG_NAMESPACE_BEGIN
class JITSessionCPU;

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<JITSession> jit;
  std::unique_ptr<llvm::Module> runtime_module, struct_module;
  JITModule *runtime_jit_module;
  Arch arch;

  SNodeAttributes snode_attr;

  TaichiLLVMContext(Arch arch);

  std::unique_ptr<llvm::Module> get_init_module();

  std::unique_ptr<llvm::Module> clone_struct_module();

  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  JITModule *add_module(std::unique_ptr<llvm::Module> module);

  virtual void *lookup_function_pointer(const std::string &name) {
    return jit->lookup(name);
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
    return TaichiLLVMContext::get_data_type(taichi::lang::get_data_type<T>());
  }

  std::size_t get_type_size(llvm::Type *type);

  template <typename T>
  llvm::Value *get_constant(T t);

  template <typename T>
  llvm::Value *get_constant(DataType dt, T t);

  llvm::DataLayout get_data_layout();

  std::string type_name(llvm::Type *type);

  void link_module_with_cuda_libdevice(std::unique_ptr<llvm::Module> &module);

  static void force_inline(llvm::Function *func);

  static void print_huge_functions(llvm::Module *module);

  static int num_instructions(llvm::Function *func);

  // remove all functions that are not (directly & indirectly) used by those
  // with export_indicator(func_name) = true
  static void eliminate_unused_functions(
      llvm::Module *module,
      std::function<bool(const std::string &)> export_indicator);

  void mark_function_as_cuda_kernel(llvm::Function *func);

  virtual ~TaichiLLVMContext();
};

TLANG_NAMESPACE_END
