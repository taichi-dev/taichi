#pragma once
#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_caching_allocator.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/llvm/llvm_device.h"

namespace taichi {
namespace lang {
namespace cuda {

class CudaResourceBinder : public ResourceBinder {
 public:
  ~CudaResourceBinder() override {
  }

  std::unique_ptr<Bindings> materialize() override{TI_NOT_IMPLEMENTED};

  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override{TI_NOT_IMPLEMENTED};
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};

  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override{TI_NOT_IMPLEMENTED};
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override{
      TI_NOT_IMPLEMENTED};
};

class CudaPipeline : public Pipeline {
 public:
  ~CudaPipeline() override {
  }

  ResourceBinder *resource_binder() override{TI_NOT_IMPLEMENTED};
};

class CudaCommandList : public CommandList {
 public:
  ~CudaCommandList() override {
  }

  void bind_pipeline(Pipeline *p) override{TI_NOT_IMPLEMENTED};
  void bind_resources(ResourceBinder *binder) override{TI_NOT_IMPLEMENTED};
  void bind_resources(ResourceBinder *binder,
                      ResourceBinder::Bindings *bindings) override{
      TI_NOT_IMPLEMENTED};
  void buffer_barrier(DevicePtr ptr, size_t size) override{TI_NOT_IMPLEMENTED};
  void buffer_barrier(DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};
  void memory_barrier() override{TI_NOT_IMPLEMENTED};
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override{
      TI_NOT_IMPLEMENTED};
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override{
      TI_NOT_IMPLEMENTED};
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override{
      TI_NOT_IMPLEMENTED};
};

class CudaStream : public Stream {
 public:
  ~CudaStream() override{};

  std::unique_ptr<CommandList> new_command_list() override{TI_NOT_IMPLEMENTED};
  StreamSemaphore submit(CommandList *cmdlist,
                         const std::vector<StreamSemaphore> &wait_semaphores =
                             {}) override{TI_NOT_IMPLEMENTED};
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override{
      TI_NOT_IMPLEMENTED};

  void command_sync() override{TI_NOT_IMPLEMENTED};
};

class CudaDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    /* Note: Memory allocation in CUDA device.
     * CudaDevice can use either its own cuda malloc mechanism via
     * `allocate_memory` or the preallocated memory managed by Llvmprogramimpl
     * via `allocate_memory_runtime`. The `use_preallocated` is used to track
     * this option. For now, we keep both options and the preallocated method is
     * used by default for CUDA backend. The `use_cached` is to enable/disable
     * the caching behavior in `allocate_memory_runtime`. Later it should be
     * always enabled, for now we keep both options to allow a scenario when
     * using preallocated memory while disabling the caching behavior.
     * */
    bool use_preallocated{true};
    bool use_cached{false};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  ~CudaDevice() override{};

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override{TI_NOT_IMPLEMENTED};

  uint64 fetch_result_uint64(int i, uint64 *result_buffer) override;

  void *map_range(DevicePtr ptr, uint64_t size) override{TI_NOT_IMPLEMENTED};
  void *map(DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};

  void unmap(DevicePtr ptr) override{TI_NOT_IMPLEMENTED};
  void unmap(DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override{
      TI_NOT_IMPLEMENTED};

  DeviceAllocation import_memory(void *ptr, size_t size);

  Stream *get_compute_stream() override{TI_NOT_IMPLEMENTED};

  void wait_idle() override{TI_NOT_IMPLEMENTED};

 private:
  std::vector<AllocInfo> allocations_;
  void validate_device_alloc(const DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      TI_ERROR("invalid DeviceAllocation");
    }
  }
  std::unique_ptr<CudaCachingAllocator> caching_allocator_{nullptr};
};

}  // namespace cuda

}  // namespace lang

}  // namespace taichi
