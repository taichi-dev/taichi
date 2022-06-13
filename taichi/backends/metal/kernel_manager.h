#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/backends/device.h"
#include "taichi/backends/metal/kernel_utils.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/lang_util.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/system/memory_pool.h"

namespace taichi {
namespace lang {

class Kernel;
struct RuntimeContext;

namespace metal {

// MetalRuntime manages everything the Metal kernels need at runtime, including
// the compiled Metal kernels pipelines and Metal buffers memory. It is the
// runtime interface between Taichi and Metal, and knows how to locate the
// series of Metal kernels generated from a Taichi kernel.
class KernelManager {
 public:
  struct Params {
    CompiledRuntimeModule compiled_runtime_module;
    CompileConfig *config;
    uint64_t *host_result_buffer;
    MemoryPool *mem_pool;
    KernelProfilerBase *profiler;
  };

  explicit KernelManager(Params params);
  // To make Pimpl + std::unique_ptr work
  ~KernelManager();

  void add_compiled_snode_tree(const CompiledStructs &snode_tree);

  // Register a Taichi kernel to the Metal runtime.
  // * |mtl_kernel_source_code| is the complete source code compiled from a
  // Taichi kernel. It may include one or more Metal compute kernels. Each
  // Metal kernel is identified by one item in |kernels_attribs|.
  //
  // TODO(k-ye): Remove |taichi_kernel_name| now that it's part of
  // |ti_kernel_attribs|. Return a handle that will be passed to
  // launch_taichi_kernel(), instead of using kernel name as the identifier.
  void register_taichi_kernel(const std::string &taichi_kernel_name,
                              const std::string &mtl_kernel_source_code,
                              const TaichiKernelAttributes &ti_kernel_attribs,
                              const KernelContextAttributes &ctx_attribs,
                              const Kernel *kernel);

  // Launch the given |taichi_kernel_name|.
  // Kernel launching is asynchronous, therefore the Metal memory is not valid
  // to access until after a synchronize() call.
  void launch_taichi_kernel(const std::string &taichi_kernel_name,
                            RuntimeContext *ctx);

  // Synchronize the memory content from Metal to host (x86_64).
  void synchronize();

  BufferMetaData get_buffer_meta_data();

  PrintStringTable *print_strtable();

  // For debugging purpose
  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  // FIXME(k-ye): This is a temporary workaround since Metal has not switched to
  // Unified Device API yet.
  DeviceAllocation allocate_memory(const Device::AllocParams &params);

 private:
  // Use Pimpl so that we can expose this interface without conditionally
  // compiling on TI_PLATFORM_OSX
  class Impl;
  std::unique_ptr<Impl> impl_{nullptr};
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
