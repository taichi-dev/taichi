#pragma once

#include <cstddef>
#include <memory>

#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/system/memory_pool.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/parallel_executor.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace llvm {
class Module;
}  // namespace llvm

namespace taichi {
namespace lang {

class StructCompiler;
class Program;

namespace cuda {
class CudaDevice;
}  // namespace cuda

namespace cpu {
class CpuDevice;
}  // namespace cpu

class LlvmProgramImpl : public ProgramImpl {
 public:
  LlvmProgramImpl(CompileConfig &config, KernelProfilerBase *profiler);

  /* ------------------------------------ */
  /* ---- JIT-Compilation Interfaces ---- */
  /* ------------------------------------ */

  // TODO(zhanlue): compile-time runtime split for LLVM::CodeGen
  // For now, compile = codegen + convert
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  void compile_snode_tree_types(SNodeTree *tree) override;

  // TODO(zhanlue): refactor materialize_snode_tree()
  // materialize_snode_tree = compile_snode_tree_types +
  // initialize_llvm_runtime_snodes It's a 2-in-1 interface
  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  void cache_kernel(const std::string &kernel_key,
                    const std::vector<LLVMCompiledData> &data_list,
                    std::vector<LlvmLaunchArgInfo> &&args);
  ;

  void cache_field(int snode_tree_id,
                   int root_id,
                   const StructCompiler &struct_compiler);

  LlvmOfflineCache::FieldCacheData get_cached_field(int snode_tree_id) const {
    TI_ASSERT(cache_data_->fields.find(snode_tree_id) !=
              cache_data_->fields.end());
    return cache_data_->fields.at(snode_tree_id);
  }

 private:
  std::unique_ptr<llvm::Module> clone_struct_compiler_initial_context(
      bool has_multiple_snode_trees,
      TaichiLLVMContext *tlctx);

  std::unique_ptr<StructCompiler> compile_snode_tree_types_impl(
      SNodeTree *tree);

  std::unique_ptr<aot::Kernel> make_aot_kernel(Kernel &kernel) override;

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  void dump_cache_data_to_disk() override;

  /* -------------------------------- */
  /* ---- JIT-Runtime Interfaces ---- */
  /* -------------------------------- */
  // ** Please implement new runtime interfaces in LlvmRuntimeExecutor **
  //
  // There are two major customer-level classes, namely Kernel and
  // FieldsBuilder.
  //
  // For now, both Kernel and FieldsBuilder rely on Program/ProgramImpl to
  // access compile time and runtime interfaces.
  //
  // We keep these runtime interfaces in ProgramImpl for now, so as to avoid
  // changing the higher-level architecture, which is coupled with base classes
  // and other backends.
  //
  // The runtime interfaces in ProgramImpl should be nothing but a simple
  // wrapper. The one with actual implementation should go inside
  // LlvmRuntimeExecutor class.

 public:
  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override {
    runtime_exec_->materialize_runtime(memory_pool, profiler,
                                       result_buffer_ptr);
  }

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    return runtime_exec_->destroy_snode_tree(snode_tree);
  }

  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return runtime_exec_->fetch_result<T>(i, result_buffer);
  }

  void finalize() override {
    runtime_exec_->finalize();
  }

  uint64_t *get_ndarray_alloc_info_ptr(const DeviceAllocation &alloc) override {
    return runtime_exec_->get_ndarray_alloc_info_ptr(alloc);
  }

  void fill_ndarray(const DeviceAllocation &alloc,
                    std::size_t size,
                    uint32_t data) override {
    return runtime_exec_->fill_ndarray(alloc, size, data);
  }

  void prepare_runtime_context(RuntimeContext *ctx) override {
    runtime_exec_->prepare_runtime_context(ctx);
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override {
    return runtime_exec_->allocate_memory_ndarray(alloc_size, result_buffer);
  }

  Device *get_compute_device() override {
    return runtime_exec_->get_compute_device();
  }

  /**
   * Initializes the SNodes for LLVM based backends.
   */
  void initialize_llvm_runtime_snodes(
      const LlvmOfflineCache::FieldCacheData &field_cache_data,
      uint64 *result_buffer) {
    runtime_exec_->initialize_llvm_runtime_snodes(field_cache_data,
                                                  result_buffer);
  }

  void initialize_host() {
    runtime_exec_->initialize_host();
  }

  void maybe_initialize_cuda_llvm_context() {
    runtime_exec_->maybe_initialize_cuda_llvm_context();
  }

  uint64 fetch_result_uint64(int i, uint64 *result_buffer) override {
    return runtime_exec_->fetch_result_uint64(i, result_buffer);
  }

  template <typename T, typename... Args>
  T runtime_query(const std::string &key,
                  uint64 *result_buffer,
                  Args &&...args) {
    return runtime_exec_->runtime_query<T>(key, result_buffer,
                                           std::forward<Args>(args)...);
  }

  void print_list_manager_info(void *list_manager, uint64 *result_buffer) {
    runtime_exec_->print_list_manager_info(list_manager, result_buffer);
  }

  void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer) override {
    runtime_exec_->print_memory_profiler_info(snode_trees_, result_buffer);
  }

  TaichiLLVMContext *get_llvm_context(Arch arch) {
    return runtime_exec_->get_llvm_context(arch);
  }

  void synchronize() override {
    runtime_exec_->synchronize();
  }

  LLVMRuntime *get_llvm_runtime() {
    return runtime_exec_->get_llvm_runtime();
  }

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return runtime_exec_->get_snode_num_dynamically_allocated(snode,
                                                              result_buffer);
  }

  void check_runtime_error(uint64 *result_buffer) override {
    runtime_exec_->check_runtime_error(result_buffer);
  }

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) override {
    // FIXME: Compute the proper offset. Current method taken from GGUI code
    size_t offset = 0;

    SNode *dense_parent = child->parent;
    SNode *root = dense_parent->parent;

    int child_id = root->child_id(dense_parent);

    for (int i = 0; i < child_id; ++i) {
      SNode *child = root->ch[i].get();
      offset += child->cell_size_bytes * child->num_cells_per_container;
    }

    return offset;
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return runtime_exec_->get_snode_tree_device_ptr(tree_id);
  }

  cuda::CudaDevice *cuda_device() {
    return runtime_exec_->cuda_device();
  }

  cpu::CpuDevice *cpu_device() {
    return runtime_exec_->cpu_device();
  }

  LlvmDevice *llvm_device() {
    return runtime_exec_->llvm_device();
  }

  LlvmRuntimeExecutor *get_runtime_executor() {
    return runtime_exec_.get();
  }

  const std::unique_ptr<LlvmOfflineCacheFileReader> &get_cache_reader() {
    return cache_reader_;
  }

  // TODO(zhanlue): Rearrange llvm::Context's ownership
  //
  // In LLVM backend, most of the compiled information are stored in
  // llvm::Module:
  // 1. Runtime functions are compiled into runtime_module,
  // 2. Fields are compiled into struct_module,
  // 3. Each kernel is compiled into individual kernel_module
  //
  // However, all the llvm::Modules are owned by llvm::Context, which belongs to
  // TaichiLLVMContext. Upon destruction, there's an implicit requirement that
  // TaichiLLVMContext has to stay alive until all the llvm::Modules are
  // destructed, otherwise there will be risks of dangling references.
  //
  // To guarantee the life cycle of llvm::Module stay aligned with
  // llvm::Context, we better make llvm::Context a more global-scoped variable,
  // instead of owned by TaichiLLVMContext.
  //
  // Objects owning llvm::Module so far (from direct to indirect):
  // 1. LlvmOfflineCache::CachedKernelData(direct owner)
  // 2. LlvmOfflineCache
  //   3.1 LlvmProgramImpl
  //   3.2 LlvmAotModuleBuilder
  //   3.3 llvm_aot::KernelImpl (for use in CGraph)
  //
  // Objects owning llvm::Context (from direct to indirect)
  // 1. TaichiLLVMContext
  // 2. LlvmProgramImpl
  //
  // Make sure the above mentioned objects are destructed in order.
  ~LlvmProgramImpl() override {
    // Explicitly enforce "LlvmOfflineCache::CachedKernelData::owned_module"
    // destructs before
    // "LlvmRuntimeExecutor::TaichiLLVMContext::ThreadSafeContext"

    // 1. Destructs cache_data_
    cache_data_.reset();

    // 2. Destructs cache_reader_
    cache_reader_.reset();

    // 3. Destructs runtime_exec_
    runtime_exec_.reset();
  }
  ParallelExecutor compilation_workers;  // parallel compilation

 private:
  std::size_t num_snode_trees_processed_{0};
  std::unique_ptr<LlvmRuntimeExecutor> runtime_exec_;
  std::unique_ptr<LlvmOfflineCache> cache_data_;
  std::unique_ptr<LlvmOfflineCacheFileReader> cache_reader_;
};

LlvmProgramImpl *get_llvm_program(Program *prog);

}  // namespace lang
}  // namespace taichi
