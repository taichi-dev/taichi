#pragma once

// llvm backend compiler (x64, arm64, cuda, amdgpu etc)
// in charge of creating & JITing arch-specific LLVM modules,
// and invoking compiled functions (kernels).
// Designed to be multithreaded for parallel compilation.

#include <mutex>
#include <functional>
#include <thread>

#include "taichi/util/lang_util.h"
#include "taichi/runtime/llvm/llvm_fwd.h"
#include "taichi/ir/snode.h"
#include "taichi/jit/jit_session.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"

namespace taichi::lang {

class JITSessionCPU;
class LlvmProgramImpl;

/**
 * Manages an LLVMContext for Taichi's usage.
 */
class TaichiLLVMContext {
 private:
  struct ThreadLocalData {
    std::unique_ptr<llvm::orc::ThreadSafeContext> thread_safe_llvm_context{
        nullptr};
    llvm::LLVMContext *llvm_context{nullptr};
    std::unique_ptr<llvm::Module> runtime_module{nullptr};
    std::unordered_map<int, std::unique_ptr<llvm::Module>> struct_modules;
    explicit ThreadLocalData(std::unique_ptr<llvm::orc::ThreadSafeContext> ctx);
    ~ThreadLocalData();
  };
  const CompileConfig &config_;

 public:
  // main_thread is defined to be the thread that runs the initializer

  std::unique_ptr<ThreadLocalData> linking_context_data{nullptr};

  TaichiLLVMContext(const CompileConfig &config, Arch arch);

  virtual ~TaichiLLVMContext();

  llvm::LLVMContext *get_this_thread_context();

  llvm::orc::ThreadSafeContext *get_this_thread_thread_safe_context();

  /**
   * Updates the LLVM module of the JIT compiled SNode structs.
   *
   * @param module Module containing the JIT compiled SNode structs.
   */
  void add_struct_module(std::unique_ptr<llvm::Module> module, int tree_id);

  void init_runtime_module(llvm::Module *runtime_module);

  /**
   * Clones the LLVM module compiled from llvm/runtime.cpp
   *
   * @return The cloned module.
   */
  std::unique_ptr<llvm::Module> clone_runtime_module();

  std::unique_ptr<llvm::Module> module_from_file(const std::string &file);

  llvm::Type *get_data_type(DataType dt);

  template <typename T>
  llvm::Type *get_data_type() {
    return TaichiLLVMContext::get_data_type(taichi::lang::get_data_type<T>());
  }

  std::size_t get_type_size(llvm::Type *type);

  std::size_t get_struct_element_offset(llvm::StructType *type, int idx);

  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(
      const StructType *old_ty,
      const std::string &layout);

  template <typename T>
  llvm::Value *get_constant(T t);

  template <typename T>
  llvm::Value *get_constant(DataType dt, T t);

  llvm::DataLayout get_data_layout();

  std::string get_data_layout_string();

  std::string type_name(llvm::Type *type);

  static void mark_inline(llvm::Function *func);

  static void print_huge_functions(llvm::Module *module);

  // remove all functions that are not (directly & indirectly) used by those
  // with export_indicator(func_name) = true
  static void eliminate_unused_functions(
      llvm::Module *module,
      std::function<bool(const std::string &)> export_indicator);

  void mark_function_as_cuda_kernel(llvm::Function *func, int block_dim = 0);

  void mark_function_as_amdgpu_kernel(llvm::Function *func);

  void fetch_this_thread_struct_module();
  llvm::Module *get_this_thread_runtime_module();
  llvm::Function *get_runtime_function(const std::string &name);
  llvm::Function *get_struct_function(const std::string &name, int tree_id);
  llvm::Type *get_runtime_type(const std::string &name);

  std::unique_ptr<llvm::Module> new_module(
      std::string name,
      llvm::LLVMContext *context = nullptr);

  void delete_snode_tree(int id);

  void add_struct_for_func(llvm::Module *module, int tls_size);

  static std::string get_struct_for_func_name(int tls_size);

  LLVMCompiledKernel link_compiled_tasks(
      std::vector<std::unique_ptr<LLVMCompiledTask>> data_list);

  static llvm::DataLayout get_data_layout(Arch arch);

 private:
  std::unique_ptr<llvm::Module> clone_module_to_context(
      llvm::Module *module,
      llvm::LLVMContext *target_context);

  void link_module_with_custom_cuda_library(
      std::unique_ptr<llvm::Module> &module);

  void link_module_with_cuda_libdevice(std::unique_ptr<llvm::Module> &module);

  void link_module_with_amdgpu_libdevice(std::unique_ptr<llvm::Module> &module);

  static int num_instructions(llvm::Function *func);

  void insert_nvvm_annotation(llvm::Function *func, std::string key, int val);

  std::unique_ptr<llvm::Module> clone_module_to_this_thread_context(
      llvm::Module *module);

  ThreadLocalData *get_this_thread_data();

  std::unordered_map<std::thread::id, std::unique_ptr<ThreadLocalData>>
      per_thread_data_;

  Arch arch_;
  llvm::DataLayout data_layout_{""};
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

}  // namespace taichi::lang
