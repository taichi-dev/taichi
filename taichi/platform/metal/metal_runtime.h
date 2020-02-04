#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "metal_api.h"
#include "metal_kernel_util.h"
#include <taichi/memory_pool.h>
#include <taichi/profiler.h>

#ifdef TC_SUPPORTS_METAL

TLANG_NAMESPACE_BEGIN

struct Context;

namespace metal {

// This class requests the Metal buffer memory of |size| bytes from |mem_pool|.
// Once allocated, it does not own the memory (hence the name "view"). Instead,
// GC is deferred to the memory pool.
class BufferMemoryView {
 public:
  BufferMemoryView(size_t size, MemoryPool *mem_pool);

  inline size_t size() const {
    return size_;
  }
  inline void *ptr() const {
    return ptr_;
  }

 private:
  size_t size_;
  void *ptr_;
};

// MetalRuntime manages everything the Metal kernels need at runtime, including
// the compiled Metal kernels pipelines and Metal buffers memory. It is the
// runtime interface between Taichi and Metal, and knows how to locate the
// series of Metal kernels generated from a Taichi kernel.
class MetalRuntime {
 public:
  MetalRuntime(size_t root_size, MemoryPool *mem_pool, CPUProfiler *profiler);

  // Register a Taichi kernel to the Metal runtime.
  // * |mtl_kernel_source_code| is the complete source code compiled from a
  // Taichi kernel. It may include one or more Metal compute kernels. Each
  // Metal kernel is identified by one item in |kernels_attribs|.
  // * |global_tmps_size| is the total size of global temporary variables,
  // computed during the offloading pass.
  void register_taichi_kernel(
      const std::string &taichi_kernel_name,
      const std::string &mtl_kernel_source_code,
      const std::vector<MetalKernelAttributes> &kernels_attribs,
      size_t global_tmps_size,
      const MetalKernelArgsAttributes &args_attribs);

  // Launch the given |taichi_kernel_name|.
  // Kernel launching is asynchronous, therefore the Metal memory is not valid
  // to access until after a synchronize() call.
  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            Context *ctx);

  // Synchronize the memory content from Metal to host (x86_64).
  void synchronize();

 private:
  void create_new_command_buffer();

  class HostMetalArgsBlitter;
  // Info for launching a compiled Metal kernel
  class CompiledMtlKernel {
   public:
    CompiledMtlKernel(const MetalKernelAttributes &md,
                      MTLDevice *device,
                      MTLFunction *func,
                      CPUProfiler *profiler);

    void launch(MTLBuffer *root_buffer,
                MTLBuffer *global_tmp_buffer,
                MTLBuffer *args_buffer,
                MTLCommandBuffer *command_buffer);

    inline MetalKernelAttributes *kernel_attribs() {
      return &kernel_attribs_;
    }

   private:
    MetalKernelAttributes kernel_attribs_;
    nsobj_unique_ptr<MTLComputePipelineState> pipeline_state_{nullptr};
    CPUProfiler *const profiler_;
    const std::string profiler_id_;
  };

  // Info for launching a compiled Taichi kernel, which consists of a series of
  // compiled Metal kernels.
  class CompiledTaichiKernel {
   public:
    CompiledTaichiKernel(
        const std::string &taichi_kernel_name,
        const std::string &source_code,
        const std::vector<MetalKernelAttributes> &mtl_kernels_attribs,
        size_t global_tmps_size,
        const MetalKernelArgsAttributes &args_attribs,
        MTLDevice *device,
        MemoryPool *mem_pool,
        CPUProfiler *profiler);

   private:
    friend void MetalRuntime::launch_taichi_kernel(
        const std::string &taichi_kernel_name,
        Context *ctx);
    friend class HostMetalArgsBlitter;

    std::string mtl_source_code_;
    std::vector<std::unique_ptr<CompiledMtlKernel>> compiled_mtl_kernels_;
    BufferMemoryView global_tmps_mem_;
    nsobj_unique_ptr<MTLBuffer> global_tmps_buffer_;
    MetalKernelArgsAttributes args_attribs_;
    std::unique_ptr<BufferMemoryView> args_mem_{nullptr};
    nsobj_unique_ptr<MTLBuffer> args_buffer_{nullptr};
    CPUProfiler *const profiler_;
  };

  MemoryPool *const mem_pool_;
  CPUProfiler *const profiler_;
  BufferMemoryView root_buffer_mem_;
  nsobj_unique_ptr<MTLDevice> device_{nullptr};
  nsobj_unique_ptr<MTLCommandQueue> command_queue_{nullptr};
  nsobj_unique_ptr<MTLCommandBuffer> cur_command_buffer_{nullptr};
  nsobj_unique_ptr<MTLBuffer> root_buffer_{nullptr};
  std::unordered_map<std::string, std::unique_ptr<CompiledTaichiKernel>>
      compiled_taichi_kernels_;
};

}  // namespace metal
TLANG_NAMESPACE_END

#endif  // TC_SUPPORTS_METAL
