#pragma once

#include <cstddef>
#include <memory>

#include "taichi/llvm/llvm_device.h"
#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/llvm/snode_tree_buffer_manager.h"
#include "taichi/inc/constants.h"
#include "taichi/program/compile_config.h"
#include "taichi/common/logging.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/llvm/launch_arg_info.h"
#include "taichi/runtime/runtime.h"
#include "taichi/system/threading.h"
#include "taichi/struct/struct.h"
#include "taichi/struct/struct_llvm.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/system/memory_pool.h"
#include "taichi/program/program_impl.h"
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

class LlvmRuntimeExecutor {
 public:
  LlvmRuntimeExecutor(CompileConfig &config);

  TaichiLLVMContext *get_llvm_context(Arch arch);

  template <typename T, typename... Args>
  T runtime_query(const std::string &key, uint64 *result_buffer, Args... args) {
    TI_ASSERT(arch_uses_llvm(config_->arch));

    TaichiLLVMContext *tlctx = nullptr;
    if (llvm_context_device_) {
      tlctx = llvm_context_device_.get();
    } else {
      tlctx = llvm_context_host_.get();
    }

    auto runtime = tlctx->runtime_jit_module;
    runtime->call<void *, Args...>("runtime_" + key, llvm_runtime_,
                                   std::forward<Args>(args)...);
    return taichi_union_cast_with_different_sizes<T>(fetch_result_uint64(
        taichi_result_buffer_runtime_query_id, result_buffer));
  }

  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return taichi_union_cast_with_different_sizes<T>(
        fetch_result_uint64(i, result_buffer));
  }

  void print_list_manager_info(void *list_manager, uint64 *result_buffer);

  void synchronize();

  LLVMRuntime *get_llvm_runtime() {
    return static_cast<LLVMRuntime *>(llvm_runtime_);
  }

  void check_runtime_error(uint64 *result_buffer);

  void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer);

  DevicePtr get_snode_tree_device_ptr(int tree_id);

  void initialize_llvm_runtime_snodes(
      const LlvmOfflineCache::FieldCacheData &field_cache_data,
      uint64 *result_buffer);

 private:
  uint64 fetch_result_uint64(int i, uint64 *result_buffer);

  void initialize_host();

  /**
   * Initializes Program#llvm_context_device, if this has not been done.
   *
   * Not thread safe.
   */
  void maybe_initialize_cuda_llvm_context();

  std::size_t get_snode_num_dynamically_allocated(SNode *snode,
                                                  uint64 *result_buffer);

  cuda::CudaDevice *cuda_device();
  cpu::CpuDevice *cpu_device();
  LlvmDevice *llvm_device();

  Device *get_compute_device() {
    return device_.get();
  }

  void destroy_snode_tree(SNodeTree *snode_tree) {
    get_llvm_context(host_arch())
        ->delete_functions_of_snode_tree(snode_tree->id());
    snode_tree_buffer_manager_->destroy(snode_tree);
  }

 private:
  CompileConfig *config_;
  std::unique_ptr<TaichiLLVMContext> llvm_context_host_{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_device_{nullptr};
  std::unordered_map<int, DeviceAllocation> snode_tree_allocs_;
  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager_{nullptr};
  void *llvm_runtime_{nullptr};

  std::shared_ptr<Device> device_{nullptr};

  // good buddy
  friend LlvmProgramImpl;
};

class LlvmProgramImpl : public ProgramImpl {
 public:
  LlvmProgramImpl(CompileConfig &config, KernelProfilerBase *profiler);

  void prepare_runtime_context(RuntimeContext *ctx) override {
    ctx->runtime = get_llvm_runtime();
  }

  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  void compile_snode_tree_types(SNodeTree *tree) override;

  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return runtime_exec_->fetch_result<T>(i, result_buffer);
  }

  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    return runtime_exec_->destroy_snode_tree(snode_tree);
  }

  void finalize();

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;

  uint64_t *get_ndarray_alloc_info_ptr(const DeviceAllocation &alloc) override;

  void fill_ndarray(const DeviceAllocation &alloc,
                    std::size_t size,
                    uint32_t data) override;

  void cache_kernel(const std::string &kernel_key,
                    llvm::Module *module,
                    std::vector<LlvmLaunchArgInfo> &&args,
                    std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
                        &&offloaded_task_list);

  void cache_field(int snode_tree_id,
                   int root_id,
                   const StructCompiler &struct_compiler);

  LlvmOfflineCache::FieldCacheData get_cached_field(int snode_tree_id) const {
    TI_ASSERT(cache_data_.fields.find(snode_tree_id) !=
              cache_data_.fields.end());
    return cache_data_.fields.at(snode_tree_id);
  }

 private:
  std::unique_ptr<llvm::Module> clone_struct_compiler_initial_context(
      bool has_multiple_snode_trees,
      TaichiLLVMContext *tlctx);

  std::unique_ptr<StructCompiler> compile_snode_tree_types_impl(
      SNodeTree *tree);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  void dump_cache_data_to_disk() override;

  /* -------------------------------- */
  /* ---- JIT-Runtime Interfaces ---- */
  /* -------------------------------- */
 public:
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

  template <typename T, typename... Args>
  T runtime_query(const std::string &key, uint64 *result_buffer, Args... args) {
    return runtime_exec_->runtime_query<T>(key, result_buffer,
                                           std::forward<Args>(args)...);
  }

  void print_list_manager_info(void *list_manager, uint64 *result_buffer) {
    runtime_exec_->print_list_manager_info(list_manager, result_buffer);
  }

  void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer) {
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

  void check_runtime_error(uint64 *result_buffer) {
    runtime_exec_->check_runtime_error(result_buffer);
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

 private:
  std::size_t num_snode_trees_processed_{0};
  LlvmOfflineCache cache_data_;
  std::unique_ptr<LlvmRuntimeExecutor> runtime_exec_;

  /* ------- Runtime: move to LlvmRuntimeExecutor --------- */
  std::unique_ptr<ThreadPool> thread_pool_{nullptr};
  std::unique_ptr<Runtime> runtime_mem_info_{nullptr};

  void *preallocated_device_buffer_{nullptr};  // TODO: move to memory allocator

  DeviceAllocation preallocated_device_buffer_alloc_{kDeviceNullAllocation};
};

LlvmProgramImpl *get_llvm_program(Program *prog);

}  // namespace lang
}  // namespace taichi
