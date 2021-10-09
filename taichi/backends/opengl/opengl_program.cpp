#include "taichi/backends/vulkan/codegen_vulkan.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"

#include "taichi/backends/opengl/opengl_program.h"
#include "taichi/backends/opengl/opengl_device.h"

namespace taichi {
namespace lang {

FunctionType OpenglProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(kernel, runtime_.get());
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  TI_ASSERT(opengl::is_opengl_api_available());

  device_ = opengl::get_opengl_device();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
}

void OpenglProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    uint64 *result_buffer) {
  runtime_->materialize_snode_tree(tree);
}

OpenglProgramImpl::~OpenglProgramImpl() {
  runtime_.reset();
  device_.reset();
}

}  // namespace lang
}  // namespace taichi
