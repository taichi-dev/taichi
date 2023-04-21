#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/rhi/llvm/llvm_device.h"

namespace taichi::lang {
namespace cpu {

class CpuPipeline : public Pipeline {
 public:
  ~CpuPipeline() override {
  }
};

class CpuCommandList : public CommandList {
 public:
  ~CpuCommandList() override {
  }

  void bind_pipeline(Pipeline *p) noexcept override{TI_NOT_IMPLEMENTED};
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept override{
      TI_NOT_IMPLEMENTED};
  RhiResult bind_raster_resources(RasterResources *res) noexcept override{
      TI_NOT_IMPLEMENTED};
  void buffer_barrier(DevicePtr ptr,
                      size_t size) noexcept override{TI_NOT_IMPLEMENTED};
  void buffer_barrier(DeviceAllocation alloc) noexcept override{
      TI_NOT_IMPLEMENTED};
  void memory_barrier() noexcept override{TI_NOT_IMPLEMENTED};
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept override{
      TI_NOT_IMPLEMENTED};
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept override{
      TI_NOT_IMPLEMENTED};
  RhiResult dispatch(uint32_t x,
                     uint32_t y = 1,
                     uint32_t z = 1) noexcept override{TI_NOT_IMPLEMENTED};
};

class CpuStream : public Stream {
 public:
  ~CpuStream() override{};

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept override{
      TI_NOT_IMPLEMENTED};
  StreamSemaphore submit(CommandList *cmdlist,
                         const std::vector<StreamSemaphore> &wait_semaphores =
                             {}) override{TI_NOT_IMPLEMENTED};
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override{
      TI_NOT_IMPLEMENTED};

  void command_sync() override{TI_NOT_IMPLEMENTED};
};

class CpuDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool use_cached{false};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  CpuDevice();
  ~CpuDevice() override{};

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) override;

  RhiResult upload_data(DevicePtr *device_ptr,
                        const void **data,
                        size_t *size,
                        int num_alloc = 1) noexcept override;

  RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept override;

  ShaderResourceSet *create_resource_set() override{TI_NOT_IMPLEMENTED};

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final {
    TI_NOT_IMPLEMENTED;
  }

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) final{TI_NOT_IMPLEMENTED};
  void unmap(DeviceAllocation alloc) final;

  DeviceAllocation import_memory(void *ptr, size_t size) override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  Stream *get_compute_stream() override{TI_NOT_IMPLEMENTED};

  void wait_idle() override{TI_NOT_IMPLEMENTED};

 private:
  std::vector<AllocInfo> allocations_;

  void validate_device_alloc(const DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      TI_ERROR("invalid DeviceAllocation");
    }
  }
};

}  // namespace cpu

}  // namespace taichi::lang
