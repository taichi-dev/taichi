#include "opengl_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/kernel_compiler.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/rhi/opengl/opengl_api.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/kernel_launcher.h"

namespace taichi::lang {

OpenglProgramImpl::OpenglProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = opengl::make_opengl_device();

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void OpenglProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void OpenglProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                               uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> OpenglProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::opengl, *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::opengl, *config, caps);
  }
}

DeviceAllocation OpenglProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  DeviceAllocation alloc;
  RhiResult res = get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false},
      &alloc);
  TI_ASSERT(res == RhiResult::success);
  return alloc;
}
DeviceAllocation OpenglProgramImpl::allocate_texture(
    const ImageParams &params) {
  return runtime_->create_image(params);
}

void OpenglProgramImpl::finalize() {
  runtime_.reset();
  device_.reset();
  opengl::reset_opengl();
}

OpenglProgramImpl::~OpenglProgramImpl() {
  finalize();
}

std::unique_ptr<KernelCompiler> OpenglProgramImpl::make_kernel_compiler() {
  spirv::KernelCompiler::Config cfg;
  cfg.compiled_struct_data = runtime_ ? &snode_tree_mgr_->get_compiled_structs()
                                      : &aot_compiled_snode_structs_;
  return std::make_unique<spirv::KernelCompiler>(std::move(cfg));
}

std::unique_ptr<KernelLauncher> OpenglProgramImpl::make_kernel_launcher() {
  gfx::KernelLauncher::Config cfg;
  cfg.gfx_runtime_ = runtime_.get();
  return std::make_unique<gfx::KernelLauncher>(std::move(cfg));
}

}  // namespace taichi::lang
