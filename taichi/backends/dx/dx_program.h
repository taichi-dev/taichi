#pragma once

#ifdef TI_WITH_DX11

#include "taichi/backends/dx/dx_device.h"
#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_tree_manager.h"
#include "taichi/program/program_impl.h"

namespace taichi {
namespace lang {

class Dx11ProgramImpl : public ProgramImpl {
 public:
  Dx11ProgramImpl(CompileConfig &config);

  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;
  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;
  }
  std::unique_ptr<AotModuleBuilder> make_aot_module_builder();
  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;
  virtual void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer_ptr) override;
  virtual void destroy_snode_tree(SNodeTree *snode_tree) override;
  void synchronize() override;

 private:
  std::shared_ptr<directx11::Dx11Device> device_{nullptr};
  std::unique_ptr<vulkan::VkRuntime> runtime_{nullptr};
  std::unique_ptr<vulkan::SNodeTreeManager> snode_tree_mgr_{nullptr};
};

}  // namespace lang
}  // namespace taichi

#endif
