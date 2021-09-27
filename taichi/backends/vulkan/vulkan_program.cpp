#include "taichi/backends/vulkan/vulkan_program.h"
using namespace taichi::lang::vulkan;

namespace taichi {
namespace lang {

FunctionType VulkanProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(kernel, vulkan_runtime_.get());
}

void VulkanProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  vulkan_runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void VulkanProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    uint64 *result_buffer) {
  // TODO: support materializing multiple snode trees
  vulkan_runtime_->materialize_snode_tree(tree);
}

}  // namespace lang
}  // namespace taichi
