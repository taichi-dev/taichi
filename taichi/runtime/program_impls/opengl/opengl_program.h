#pragma once

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/program/program_impl.h"

namespace taichi::lang {

class OpenglProgramImpl : public ProgramImpl {
 public:
  explicit OpenglProgramImpl(CompileConfig &config);
  ~OpenglProgramImpl() override;
  FunctionType compile(const CompileConfig &compile_config,
                       Kernel *kernel) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support sparse
  }

  void compile_snode_tree_types(SNodeTree *tree) override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  void synchronize() override {
    runtime_->synchronize();
  }

  void finalize() override;

  StreamSemaphore flush() override {
    return runtime_->flush();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_ASSERT(snode_tree_mgr_ != nullptr);
    snode_tree_mgr_->destroy_snode_tree(snode_tree);
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;

  bool used_in_kernel(DeviceAllocationId id) override {
    return runtime_->used_in_kernel(id);
  }

  DeviceAllocation allocate_texture(const ImageParams &params) override;

  Device *get_compute_device() override {
    return device_.get();
  }

  Device *get_graphics_device() override {
    return device_.get();
  }

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) override {
    return snode_tree_mgr_->get_field_in_tree_offset(tree_id, child);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return snode_tree_mgr_->get_snode_tree_device_ptr(tree_id);
  }

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;
  std::unique_ptr<KernelLauncher> make_kernel_launcher() override;

 private:
  std::shared_ptr<Device> device_{nullptr};
  std::unique_ptr<gfx::GfxRuntime> runtime_{nullptr};
  std::unique_ptr<gfx::SNodeTreeManager> snode_tree_mgr_{nullptr};
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_;
};

}  // namespace taichi::lang
