#pragma once
#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/device.h"

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
  void submit(CommandList *cmdlist) override{TI_NOT_IMPLEMENTED};
  void submit_synced(CommandList *cmdlist) override{TI_NOT_IMPLEMENTED};

  void command_sync() override{TI_NOT_IMPLEMENTED};
};

class CudaDevice : public Device {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
  };

  AllocInfo get_alloc_info(DeviceAllocation handle);

  ~CudaDevice() override{};

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override{TI_NOT_IMPLEMENTED};

  void *map_range(DevicePtr ptr, uint64_t size) override{TI_NOT_IMPLEMENTED};
  void *map(DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};

  void unmap(DevicePtr ptr) override{TI_NOT_IMPLEMENTED};
  void unmap(DeviceAllocation alloc) override{TI_NOT_IMPLEMENTED};

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override{
      TI_NOT_IMPLEMENTED};

  DeviceAllocation import_memory(void *ptr, size_t size);

  Stream *get_compute_stream() override{TI_NOT_IMPLEMENTED};

 private:
  std::vector<AllocInfo> allocations_;
  void validate_device_alloc(DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      TI_ERROR("invalid DeviceAllocation");
    }
  }
};

}  // namespace cuda

}  // namespace lang

}  // namespace taichi
