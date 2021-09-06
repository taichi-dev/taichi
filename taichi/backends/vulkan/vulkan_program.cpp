#include "taichi/backends/vulkan/vulkan_program.h"
using namespace taichi::lang::vulkan;

namespace taichi {
namespace lang {

FunctionType VulkanProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(
      kernel, &vulkan_compiled_structs_.value(), vulkan_runtime_.get());
}

void VulkanProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  // doesn't do anything other than alloc result buffer. runtime is materialized
  // together with snode tree.
  // TODO: separate runtime materialization and tree materialization.
}

void VulkanProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    SNodeGlobalVarExprMap &,
    uint64 *result_buffer) {
  // TODO: support materializing multiple snode trees
  auto *const root = tree->root();
  vulkan_compiled_structs_ = vulkan::compile_snode_structs(*root);
  vulkan::VkRuntime::Params params;
  params.snode_descriptors = &(vulkan_compiled_structs_->snode_descriptors);
  params.host_result_buffer = result_buffer;
  vulkan_runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

}  // namespace lang
}  // namespace taichi
