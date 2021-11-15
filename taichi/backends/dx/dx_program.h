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
    return 0;  // TODO: support dynamic snode alloc in vulkan
  }

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      std::unordered_map<int, SNode *> &snodes,
      uint64 *result_buffer) override;

  void synchronize() override {
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    // TODO: implement opengl aot
    return nullptr;
  }

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_NOT_IMPLEMENTED
  }

  ~DxProgramImpl() override;

 private:
  std::unique_ptr<Device> device_;
  std::unique_ptr<vulkan::VkRuntime> runtime_;
};
}  // namespace lang
}  // namespace taichi