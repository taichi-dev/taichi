#include "taichi/backends/vulkan/codegen_vulkan.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"

#include "taichi/backends/opengl/opengl_program.h"

#ifdef TI_WITH_OPENGL
#include "taichi/backends/opengl/opengl_device.h"
#endif

namespace taichi {
namespace lang {

FunctionType OpenglProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
#ifdef TI_WITH_OPENGL
  vulkan::lower(kernel);
  return vulkan::compile_to_executable(kernel, runtime_.get());
#else
  TI_NOT_IMPLEMENTED
#endif
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
#ifdef TI_WITH_OPENGL
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  TI_ASSERT(opengl::is_opengl_api_available());

  device_ = opengl::get_opengl_device();

  vulkan::VkRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<vulkan::VkRuntime>(std::move(params));
#else
  TI_NOT_IMPLEMENTED
#endif
}

void OpenglProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) {
#ifdef TI_WITH_OPENGL
  runtime_->materialize_snode_tree(tree);
#else
  TI_NOT_IMPLEMENTED
#endif
}

OpenglProgramImpl::~OpenglProgramImpl() {
  runtime_.reset();
  device_.reset();
}

}  // namespace lang
}  // namespace taichi
