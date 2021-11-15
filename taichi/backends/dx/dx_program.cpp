#include "taichi/backends/vulkan/codegen_vulkan.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"

#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_api.h"

namespace taichi {
namespace lang {

FunctionType DxProgramImpl::compile(Kernel *kernel, OffloadedStmt *offloaded) {
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(kernel, runtime_.get());
}

void DxProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                        KernelProfilerBase *profiler,
                                        uint64 **result_buffer_ptr) {
  // 1. allocate result buffer
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  TI_ASSERT(dx::is_dx_api_available());

  device_ = dx::get_dx_device();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void DxProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    std::unordered_map<int, SNode *> &snodes,
    uint64 *result_buffer) {
  runtime_->materialize_snode_tree(tree);
}

DxProgramImpl::~DxProgramImpl() {
}

}  // namespace lang
}  // namespace taichi