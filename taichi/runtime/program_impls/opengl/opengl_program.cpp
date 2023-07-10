#include "opengl_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/kernel_compiler.h"
#include "taichi/rhi/opengl/opengl_api.h"
#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/rhi/common/host_memory_pool.h"

namespace taichi::lang {

OpenglProgramImpl::OpenglProgramImpl(CompileConfig &config)
    : GfxProgramImpl(config) {
}

void OpenglProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)HostMemoryPool::get_instance().allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = opengl::make_opengl_device();

  gfx::GfxRuntime::Params params;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void OpenglProgramImpl::finalize() {
  opengl::reset_opengl();
}

}  // namespace taichi::lang
