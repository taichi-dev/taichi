#pragma once
#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/device.h"

TLANG_NAMESPACE_BEGIN


namespace cuda{

class CudaResourceBinder:public ResourceBinder {
 public:
  virtual ~CudaResourceBinder() override {
  }

  virtual std::unique_ptr<Bindings> materialize() override {
      TI_NOT_IMPLEMENTED
  };

  virtual void rw_buffer(uint32_t set,
                         uint32_t binding,
                         DevicePtr ptr,
                         size_t size) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void rw_buffer(uint32_t set,
                         uint32_t binding,
                         DeviceAllocation alloc) override {
      TI_NOT_IMPLEMENTED
  };

  virtual void buffer(uint32_t set,
                      uint32_t binding,
                      DevicePtr ptr,
                      size_t size) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void buffer(uint32_t set,
                      uint32_t binding,
                      DeviceAllocation alloc) override {
      TI_NOT_IMPLEMENTED
  };

};



class CudaPipeline:public Pipeline {
 public:
  virtual ~CudaPipeline() override {
  }

  virtual ResourceBinder *resource_binder() override {
      TI_NOT_IMPLEMENTED
  };
};


class CudaCommandList:public CommandList {
 public:
  virtual ~CudaCommandList() override {
  }

  virtual void bind_pipeline(Pipeline *p) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void bind_resources(ResourceBinder *binder) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void bind_resources(ResourceBinder *binder,
                              ResourceBinder::Bindings *bindings) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void buffer_barrier(DevicePtr ptr, size_t size) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void buffer_barrier(DeviceAllocation alloc) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void memory_barrier() override {
      TI_NOT_IMPLEMENTED
  };
  virtual void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override {
      TI_NOT_IMPLEMENTED
  };

};


class CudaStream :public Stream{
 public:
  virtual ~CudaStream() override{};

  virtual std::unique_ptr<CommandList> new_command_list() override {
      TI_NOT_IMPLEMENTED
  };
  virtual void submit(CommandList *cmdlist) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void submit_synced(CommandList *cmdlist) override {
      TI_NOT_IMPLEMENTED
  };

  virtual void command_sync() override {
      TI_NOT_IMPLEMENTED
  };
};

class CudaDevice :public Device{
 public:

  struct AllocInfo{
    void* ptr{nullptr};
    size_t size{0};
  };

  AllocInfo get_alloc_info(DeviceAllocation handle);


  virtual ~CudaDevice() override {};

  virtual DeviceAllocation allocate_memory(const AllocParams &params) ;
  virtual void dealloc_memory(DeviceAllocation handle) ;

  virtual std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override {
      TI_NOT_IMPLEMENTED
  };

  virtual void *map_range(DevicePtr ptr, uint64_t size) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void *map(DeviceAllocation alloc) override {
      TI_NOT_IMPLEMENTED
  };

  virtual void unmap(DevicePtr ptr) override {
      TI_NOT_IMPLEMENTED
  };
  virtual void unmap(DeviceAllocation alloc) override {
      TI_NOT_IMPLEMENTED
  };
  
  virtual void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override {
      TI_NOT_IMPLEMENTED
  };

  virtual Stream *get_compute_stream() override {
      TI_NOT_IMPLEMENTED
  };
private:
  std::vector<AllocInfo> allocations_;
  void validate_device_alloc(DeviceAllocation alloc){
    if(allocations_.size() <= alloc.alloc_id){
      TI_ERROR("invalid DeviceAllocation");
    }
  }
};




}



TLANG_NAMESPACE_END