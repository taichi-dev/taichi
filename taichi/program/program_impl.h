#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/device.h"
#include "taichi/aot/graph_data.h"

namespace taichi {
namespace lang {

class ProgramImpl {
 public:
  // TODO: Make it safer, we exposed it for now as it's directly accessed
  // outside.
  CompileConfig *config;

 public:
  ProgramImpl(CompileConfig &config);

  /**
   * Codegen to specific backend
   */
  virtual FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) = 0;

  /**
   * Allocate runtime buffer, e.g result_buffer or backend specific runtime
   * buffer, e.g. preallocated_device_buffer on CUDA.
   */
  virtual void materialize_runtime(MemoryPool *memory_pool,
                                   KernelProfilerBase *profiler,
                                   uint64 **result_buffer_ptr) = 0;

  /**
   * JIT compiles @param tree to backend-specific data types.
   */
  virtual void compile_snode_tree_types(SNodeTree *tree);

  /**
   * Compiles the @param tree types and allocates runtime buffer for it.
   */
  virtual void materialize_snode_tree(SNodeTree *tree,
                                      uint64 *result_buffer_ptr) = 0;

  virtual void destroy_snode_tree(SNodeTree *snode_tree) = 0;

  virtual std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) = 0;

  /**
   * Perform a backend synchronization.
   */
  virtual void synchronize() = 0;

  virtual StreamSemaphore flush() {
    synchronize();
    return nullptr;
  }

  /**
   * Make a AotModulerBuilder, currently only supported by metal and wasm.
   */
  virtual std::unique_ptr<AotModuleBuilder> make_aot_module_builder() = 0;

  /**
   * Compile a taichi::lang::Kernel to taichi::lang::aot::Kernel.
   */
  virtual std::unique_ptr<aot::Kernel> make_aot_kernel(Kernel &kernel) {
    TI_NOT_IMPLEMENTED;
  }

  /**
   * Dump Offline-cache data to disk
   */
  virtual void dump_cache_data_to_disk() {
  }

  virtual Device *get_compute_device() {
    return nullptr;
  }

  virtual Device *get_graphics_device() {
    return nullptr;
  }

  virtual DevicePtr get_snode_tree_device_ptr(int tree_id) {
    return kDeviceNullPtr;
  }

  virtual DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                                   uint64 *result_buffer) {
    return kDeviceNullAllocation;
  }
  virtual ~ProgramImpl() {
  }

 private:
};

}  // namespace lang
}  // namespace taichi
