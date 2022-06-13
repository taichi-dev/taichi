#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/llvm/llvm_device.h"
#include "taichi/system/virtual_memory.h"

namespace taichi {
namespace lang {
namespace cpu {

class CpuResourceBinder : public ResourceBinder {
 public:
  ~CpuResourceBinder() override {
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

class CpuPipeline : public Pipeline {
 public:
  ~CpuPipeline() override {
  }

  ResourceBinder *resource_binder() override{TI_NOT_IMPLEMENTED};
};

class CpuCommandList : public CommandList {
 public:
  ~CpuCommandList() override {
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

class CpuStream : public Stream {
 public:
  ~CpuStream() override{};

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

class CpuDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool use_cached{false};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  ~CpuDevice() override{};

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

  DeviceAllocation import_memory(void *ptr, size_t size);

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override{
      TI_NOT_IMPLEMENTED};

  Stream *get_compute_stream() override{TI_NOT_IMPLEMENTED};

  void wait_idle() override{TI_NOT_IMPLEMENTED};

 private:
  std::vector<AllocInfo> allocations_;
  std::unordered_map<int, std::unique_ptr<VirtualMemoryAllocator>>
      virtual_memories_;

  void validate_device_alloc(const DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      TI_ERROR("invalid DeviceAllocation");
    }
  }
};

}  // namespace cpu

}  // namespace lang

}  // namespace taichi
