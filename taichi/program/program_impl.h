#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/program/kernel_launcher.h"
#include "taichi/rhi/device.h"
#include "taichi/aot/graph_data.h"
#include "taichi/codegen/kernel_compiler.h"
#include "taichi/compilation_manager/kernel_compilation_manager.h"

namespace taichi::lang {

// Represents an image resource reference for a compute/render Op
struct ComputeOpImageRef {
  DeviceAllocation image;
  // The requested initial layout of the image, when Op is invoked
  ImageLayout initial_layout;
  // The final layout the image will be in once Op finishes
  ImageLayout final_layout;
};

struct RuntimeContext;

class ProgramImpl {
 public:
  // TODO: Make it safer, we exposed it for now as it's directly accessed
  // outside.
  CompileConfig *config;

 public:
  explicit ProgramImpl(CompileConfig &config);

  /**
   * Allocate runtime buffer, e.g result_buffer or backend specific runtime
   * buffer, e.g. preallocated_device_buffer on CUDA.
   */
  virtual void materialize_runtime(KernelProfilerBase *profiler,
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
   * Make a AotModulerBuilder.
   */
  virtual std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) = 0;

  /**
   * Dump Offline-cache data to disk
   */
  virtual void dump_cache_data_to_disk();

  virtual Device *get_compute_device() {
    return nullptr;
  }

  virtual Device *get_graphics_device() {
    return nullptr;
  }

  virtual size_t get_field_in_tree_offset(int tree_id, const SNode *child) {
    return 0;
  }

  virtual DevicePtr get_snode_tree_device_ptr(int tree_id) {
    return kDeviceNullPtr;
  }

  virtual DeviceAllocation allocate_memory_on_device(std::size_t alloc_size,
                                                     uint64 *result_buffer) {
    return kDeviceNullAllocation;
  }

  virtual bool used_in_kernel(DeviceAllocationId) {
    return false;
  }

  virtual DeviceAllocation allocate_texture(const ImageParams &params) {
    return kDeviceNullAllocation;
  }

  virtual ~ProgramImpl() {
  }

  // TODO: Move to Runtime Object
  virtual uint64_t *get_device_alloc_info_ptr(const DeviceAllocation &alloc) {
    TI_ERROR(
        "get_device_alloc_info_ptr() not implemented on the current backend");
    return nullptr;
  }

  // TODO: Move to Runtime Object
  virtual void fill_ndarray(const DeviceAllocation &alloc,
                            std::size_t size,
                            uint32_t data) {
    TI_ERROR("fill_ndarray() not implemented on the current backend");
  }

  virtual void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void print_memory_profiler_info(
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer) {
    TI_ERROR(
        "print_memory_profiler_info() not implemented on the current backend");
  }

  virtual void check_runtime_error(uint64 *result_buffer) {
    TI_ERROR("check_runtime_error() not implemented on the current backend");
  }

  virtual void finalize() {
  }

  virtual uint64 fetch_result_uint64(int i, uint64 *result_buffer) {
    return result_buffer[i];
  }

  virtual std::string get_kernel_return_data_layout() {
    return "";
  };

  virtual std::string get_kernel_argument_data_layout() {
    return "";
  };

  virtual std::pair<const StructType *, size_t>
  get_struct_type_with_data_layout(const StructType *old_ty,
                                   const std::string &layout) {
    return {old_ty, 0};
  }

  KernelCompilationManager &get_kernel_compilation_manager();

  KernelLauncher &get_kernel_launcher();

  virtual DeviceCapabilityConfig get_device_caps() {
    return {};
  }

 protected:
  virtual std::unique_ptr<KernelCompiler> make_kernel_compiler() = 0;

  virtual std::unique_ptr<KernelLauncher> make_kernel_launcher() {
    TI_NOT_IMPLEMENTED;
  }

 private:
  std::unique_ptr<KernelCompilationManager> kernel_com_mgr_;
  std::unique_ptr<KernelLauncher> kernel_launcher_;
};

}  // namespace taichi::lang
