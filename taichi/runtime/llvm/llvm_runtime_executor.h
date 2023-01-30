#pragma once

#include <cstddef>
#include <memory>

#ifdef TI_WITH_LLVM

#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/snode_tree_buffer_manager.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/compile_config.h"

#include "taichi/system/threading.h"
#include "taichi/system/memory_pool.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

namespace cuda {
class CudaDevice;
}  // namespace cuda

namespace cpu {
class CpuDevice;
}  // namespace cpu

class LlvmRuntimeExecutor {
 public:
  LlvmRuntimeExecutor(CompileConfig &config, KernelProfilerBase *profiler);

  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr);

  // SNodeTree Allocation
  void initialize_llvm_runtime_snodes(
      const LlvmOfflineCache::FieldCacheData &field_cache_data,
      uint64 *result_buffer);

  // Ndarray Allocation
  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer);

  void deallocate_memory_ndarray(DeviceAllocation handle);

  void check_runtime_error(uint64 *result_buffer);

  uint64_t *get_ndarray_alloc_info_ptr(const DeviceAllocation &alloc);

  const CompileConfig &get_config() const {
    return config_;
  }

  TaichiLLVMContext *get_llvm_context(Arch arch);

  LLVMRuntime *get_llvm_runtime();

  void prepare_runtime_context(RuntimeContext *ctx);

  Device *get_compute_device();

  LlvmDevice *llvm_device();

  void synchronize();

 private:
  /* ----------------------- */
  /* ------ Allocation ----- */
  /* ----------------------- */
  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return taichi_union_cast_with_different_sizes<T>(
        fetch_result_uint64(i, result_buffer));
  }

  template <typename T>
  T fetch_result(char *result_buffer, int offset);

  DevicePtr get_snode_tree_device_ptr(int tree_id);

  void fill_ndarray(const DeviceAllocation &alloc,
                    std::size_t size,
                    uint32_t data);

  /* ------------------------- */
  /* ---- Runtime Helpers ---- */
  /* ------------------------- */
  void print_list_manager_info(void *list_manager, uint64 *result_buffer);
  void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer);

  template <typename T, typename... Args>
  T runtime_query(const std::string &key,
                  uint64 *result_buffer,
                  Args &&...args) {
    TI_ASSERT(arch_uses_llvm(config_.arch));

    TaichiLLVMContext *tlctx = nullptr;
    if (llvm_context_device_) {
      tlctx = llvm_context_device_.get();
    } else {
      tlctx = llvm_context_host_.get();
    }

    auto runtime = tlctx->runtime_jit_module;
    runtime->call<void *>("runtime_" + key, llvm_runtime_,
                          std::forward<Args>(args)...);
    return taichi_union_cast_with_different_sizes<T>(fetch_result_uint64(
        taichi_result_buffer_runtime_query_id, result_buffer));
  }

  /* -------------------------- */
  /* ------ Member Access ----- */
  /* -------------------------- */
  cuda::CudaDevice *cuda_device();
  cpu::CpuDevice *cpu_device();

  void initialize_host();

  /**
   * Initializes Program#llvm_context_device, if this has not been done.
   *
   * Not thread safe.
   */
  void maybe_initialize_cuda_llvm_context();

  void maybe_initialize_amdgpu_llvm_context();

  void finalize();

  uint64 fetch_result_uint64(int i, uint64 *result_buffer);
  void destroy_snode_tree(SNodeTree *snode_tree);
  std::size_t get_snode_num_dynamically_allocated(SNode *snode,
                                                  uint64 *result_buffer);

 private:
  CompileConfig &config_;

  // TODO(zhanlue): compile - runtime split for TaichiLLVMContext
  //
  // TaichiLLVMContext is a thread-safe class with llvm::Module for compilation
  // and JITSession/JITModule for runtime loading & execution
  std::unique_ptr<TaichiLLVMContext> llvm_context_host_{nullptr};
  std::unique_ptr<TaichiLLVMContext> llvm_context_device_{nullptr};
  void *llvm_runtime_{nullptr};

  std::unique_ptr<ThreadPool> thread_pool_{nullptr};
  std::shared_ptr<Device> device_{nullptr};

  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager_{nullptr};
  std::unordered_map<int, DeviceAllocation> snode_tree_allocs_;
  void *preallocated_device_buffer_{nullptr};  // TODO: move to memory allocator
  DeviceAllocation preallocated_device_buffer_alloc_{kDeviceNullAllocation};

  // good buddy
  friend LlvmProgramImpl;
  friend SNodeTreeBufferManager;

  KernelProfilerBase *profiler_ = nullptr;
};

}  // namespace taichi::lang

#endif  // TI_WITH_LLVM
