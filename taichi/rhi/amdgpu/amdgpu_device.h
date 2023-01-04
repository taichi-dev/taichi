#pragma once
#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_caching_allocator.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/llvm/llvm_device.h"

namespace taichi {
namespace lang {
namespace amdgpu {

class AmdgpuCommandList : public CommandList {
 public:
  ~AmdgpuCommandList() override {
  }

  void bind_pipeline(Pipeline *p) noexcept final{TI_NOT_IMPLEMENTED};
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final{
      TI_NOT_IMPLEMENTED};
  RhiResult bind_raster_resources(RasterResources *res) noexcept final{
      TI_NOT_IMPLEMENTED};
  void buffer_barrier(DevicePtr ptr,
                      size_t size) noexcept final{TI_NOT_IMPLEMENTED};
  void buffer_barrier(DeviceAllocation alloc) noexcept final{
      TI_NOT_IMPLEMENTED};
  void memory_barrier() noexcept final{TI_NOT_IMPLEMENTED};
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final{
      TI_NOT_IMPLEMENTED};
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final{
      TI_NOT_IMPLEMENTED};
  RhiResult dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept override{
      TI_NOT_IMPLEMENTED};
};

class AmdgpuStream : public Stream {
 public:
  ~AmdgpuStream() override{};

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

class AmdgpuDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    bool use_preallocated{true};
    bool use_cached{false};
    void *mapped{nullptr};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  ~AmdgpuDevice() override{};

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  ShaderResourceSet *create_resource_set() final{TI_NOT_IMPLEMENTED};

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override{TI_NOT_IMPLEMENTED};

  uint64 fetch_result_uint64(int i, uint64 *result_buffer) override;

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final {
    TI_NOT_IMPLEMENTED;
  }
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) override{TI_NOT_IMPLEMENTED};
  void unmap(DeviceAllocation alloc) override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

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
  std::unique_ptr<AmdgpuCachingAllocator> caching_allocator_{nullptr};
};

}  // namespace amdgpu

}  // namespace lang

}  // namespace taichi
