#pragma once

#include "taichi/backends/vulkan/runtime.h"

#include "taichi/program/program_impl.h"

namespace taichi {
namespace lang {
class DxProgramImpl : public ProgramImpl {
 public:
  DxProgramImpl(CompileConfig &config) : ProgramImpl(config) {
  }

  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support sparse in vulkan
  }

  void compile_snode_tree_types(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree,
                              std::vector<std::unique_ptr<SNodeTree>> &,
                              uint64 *result_buffer) override;

  void synchronize() override {
    runtime_->synchronize();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    TI_NOT_IMPLEMENTED;
  }

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
    runtime_->destroy_snode_tree(snode_tree);
  }

  Device *get_compute_device() override {
    if (device_) {
      return device_.get();
    }
    return nullptr;
  }

  Device *get_graphics_device() override {
    TI_NOT_IMPLEMENTED;
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return runtime_->get_snode_tree_device_ptr(tree_id);
  }

  ~DxProgramImpl() override;

 private:
  std::unique_ptr<Device> device_;
  std::unique_ptr<vulkan::VkRuntime> runtime_;
};
}  // namespace lang
}  // namespace taichi