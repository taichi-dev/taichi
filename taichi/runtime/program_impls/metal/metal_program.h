#pragma once
#include "taichi/aot/module_loader.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/rhi/metal/metal_device.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"

#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/program.h"

namespace taichi::lang {

class MetalProgramImpl : public ProgramImpl {
 public:
  explicit MetalProgramImpl(CompileConfig &config);
  FunctionType compile(const CompileConfig &compile_config,
                       Kernel *kernel) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support sparse
  }

  void compile_snode_tree_types(SNodeTree *tree) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  void synchronize() override {
    if (gfx_runtime_) {
      gfx_runtime_->synchronize();
    }
  }

  StreamSemaphore flush() override {
    return gfx_runtime_->flush();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_ASSERT(snode_tree_mgr_ != nullptr);
    snode_tree_mgr_->destroy_snode_tree(snode_tree);
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;
  DeviceAllocation allocate_texture(const ImageParams &params) override;

  bool used_in_kernel(DeviceAllocationId id) override {
    return gfx_runtime_->used_in_kernel(id);
  }

  Device *get_compute_device() override {
    if (embedded_device_) {
      return embedded_device_.get();
    }
    return nullptr;
  }

  Device *get_graphics_device() override {
    if (embedded_device_) {
      return embedded_device_.get();
    }
    return nullptr;
  }

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) override {
    return snode_tree_mgr_->get_field_in_tree_offset(tree_id, child);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return snode_tree_mgr_->get_snode_tree_device_ptr(tree_id);
  }

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;

  ~MetalProgramImpl() override;

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;

 private:
  std::unique_ptr<metal::MetalDevice> embedded_device_{nullptr};
  std::unique_ptr<gfx::GfxRuntime> gfx_runtime_{nullptr};
  std::unique_ptr<gfx::SNodeTreeManager> snode_tree_mgr_{nullptr};
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_{};
};

}  // namespace taichi::lang
