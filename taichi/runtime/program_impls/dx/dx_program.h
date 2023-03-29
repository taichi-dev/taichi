#pragma once

#ifdef TI_WITH_DX11

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/program/program_impl.h"

namespace taichi::lang {

class Dx11ProgramImpl : public ProgramImpl {
 public:
  Dx11ProgramImpl(CompileConfig &config);
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
    runtime_->synchronize();
  }

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

  // TODO: These three functions are the same in Vulkan, Metal, DX and OpenGL.
  //   We should add a GfxProgramImpl base class for these functions.
  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(
      const StructType *old_ty,
      const std::string &layout) override {
    return gfx::GfxRuntime::get_struct_type_with_data_layout(old_ty, layout);
  }

  std::string get_kernel_return_data_layout() override {
    return "r-";
  };

  std::string get_kernel_argument_data_layout() override {
    auto has_buffer_ptr = runtime_->get_ti_device()->get_caps().get(
        DeviceCapability::spirv_has_physical_storage_buffer);
    return "a" + std::string(has_buffer_ptr ? "b" : "-");
  };

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;

 private:
  std::shared_ptr<Device> device_{nullptr};
  std::unique_ptr<gfx::GfxRuntime> runtime_{nullptr};
  std::unique_ptr<gfx::SNodeTreeManager> snode_tree_mgr_{nullptr};
  std::vector<spirv::CompiledSNodeStructs> aot_compiled_snode_structs_;
};

}  // namespace taichi::lang

#endif
