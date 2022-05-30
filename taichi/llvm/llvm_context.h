#pragma once

// llvm backend compiler (x64, arm64, cuda, amdgpu etc)
// in charge of creating & JITing arch-specific LLVM modules,
// and invoking compiled functions (kernels).
// Designed to be multithreaded for parallel compilation.

#include <mutex>
#include <functional>
#include <thread>

#include "taichi/lang_util.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/ir/snode.h"
#include "taichi/jit/jit_session.h"

namespace taichi {
namespace lang {

class JITSessionCPU;

/**
 * Manages an LLVMContext for Taichi's usage.
 */
class TaichiLLVMContext {
 private:
  struct ThreadLocalData {
    llvm::LLVMContext *llvm_context{nullptr};
    std::unique_ptr<llvm::orc::ThreadSafeContext> thread_safe_llvm_context{
        nullptr};
    std::unique_ptr<llvm::Module> runtime_module{nullptr};
    std::unique_ptr<llvm::Module> struct_module{nullptr};
  };

 public:
  std::unique_ptr<JITSession> jit{nullptr};
  // main_thread is defined to be the thread that runs the initializer
  JITModule *runtime_jit_module{nullptr};

  TaichiLLVMContext(LlvmProgramImpl *llvm_prog, Arch arch);

  virtual ~TaichiLLVMContext();

  llvm::LLVMContext *get_this_thread_context();

  llvm::orc::ThreadSafeContext *get_this_thread_thread_safe_context();

  /**
   * Initializes TaichiLLVMContext#runtime_jit_module.
   *
   * Unfortuantely, this cannot be placed inside the constructor. When adding an
   * llvm::Module, the JITSessionCPU implementation eventually calls back to
   * this object, so it must be fully constructed by then.
   */
  void init_runtime_jit_module();

  /**
   * Clones the LLVM module containing the JIT compiled SNode structs.
   *
   * @return The cloned module.
   */
  std::unique_ptr<llvm::Module> clone_struct_module();

  /**
   * Updates the LLVM module of the JIT compiled SNode structs.
   *
   * @param module Module containg the JIT compiled SNode structs.
   */
  void set_struct_module(const std::unique_ptr<llvm::Module> &module);

  /**
   * Clones the LLVM module compiled from llvm/runtime.cpp
   *
   * @return The cloned module.
   */
  std::unique_ptr<llvm::Module> clone_runtime_module();

  std::unique_ptr<llvm::Module> clone_module(const std::string &file);

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

  llvm::Type *get_data_type(DataType dt);

  llvm::Module *get_this_thread_struct_module();

  template <typename T>
  llvm::Type *get_data_type() {
    return TaichiLLVMContext::get_data_type(taichi::lang::get_data_type<T>());
  }

  std::size_t get_type_size(llvm::Type *type);

  std::size_t get_struct_element_offset(llvm::StructType *type, int idx);

  template <typename T>
  llvm::Value *get_constant(T t);

  template <typename T>
  llvm::Value *get_constant(DataType dt, T t);

  llvm::DataLayout get_data_layout();

  std::string type_name(llvm::Type *type);

  static void mark_inline(llvm::Function *func);

  static void print_huge_functions(llvm::Module *module);

  // remove all functions that are not (directly & indirectly) used by those
  // with export_indicator(func_name) = true
  static void eliminate_unused_functions(
      llvm::Module *module,
      std::function<bool(const std::string &)> export_indicator);

  void mark_function_as_cuda_kernel(llvm::Function *func, int block_dim = 0);

  void add_function_to_snode_tree(int id, std::string func);

  void delete_functions_of_snode_tree(int id);

 private:
  std::unique_ptr<llvm::Module> clone_module_to_context(
      llvm::Module *module,
      llvm::LLVMContext *target_context);

  void link_module_with_cuda_libdevice(std::unique_ptr<llvm::Module> &module);

  static int num_instructions(llvm::Function *func);

  void insert_nvvm_annotation(llvm::Function *func, std::string key, int val);

  std::unique_ptr<llvm::Module> clone_module_to_this_thread_context(
      llvm::Module *module);

  ThreadLocalData *get_this_thread_data();

  void update_runtime_jit_module(std::unique_ptr<llvm::Module> module);

  std::unordered_map<std::thread::id, std::unique_ptr<ThreadLocalData>>
      per_thread_data_;

  Arch arch_;

  std::thread::id main_thread_id_;
  ThreadLocalData *main_thread_data_{nullptr};
  std::mutex mut_;
  std::mutex thread_map_mut_;

  std::unordered_map<int, std::vector<std::string>> snode_tree_funcs_;
};

class LlvmModuleBitcodeLoader {
 public:
  LlvmModuleBitcodeLoader &set_bitcode_path(const std::string &bitcode_path) {
    bitcode_path_ = bitcode_path;
    return *this;
  }

  LlvmModuleBitcodeLoader &set_buffer_id(const std::string &buffer_id) {
    buffer_id_ = buffer_id;
    return *this;
  }

  LlvmModuleBitcodeLoader &set_inline_funcs(bool inline_funcs) {
    inline_funcs_ = inline_funcs;
    return *this;
  }

  std::unique_ptr<llvm::Module> load(llvm::LLVMContext *ctx) const;

 private:
  std::string bitcode_path_;
  std::string buffer_id_;
  bool inline_funcs_{false};
};

std::unique_ptr<llvm::Module> module_from_bitcode_file(
    const std::string &bitcode_path,
    llvm::LLVMContext *ctx);

}  // namespace lang
}  // namespace taichi
