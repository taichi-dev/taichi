#include "opengl_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/rhi/opengl/opengl_api.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi::lang {

namespace {

FunctionType register_params_to_executable(
    gfx::GfxRuntime::RegisterParams &&params,
    gfx::GfxRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(std::move(params));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace

OpenglProgramImpl::OpenglProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}

FunctionType OpenglProgramImpl::compile(const CompileConfig &compile_config,
                                        Kernel *kernel) {
  return register_params_to_executable(
      get_cache_manager()->load_or_compile(compile_config, kernel),
      runtime_.get());
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
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}
DeviceAllocation OpenglProgramImpl::allocate_texture(
    const ImageParams &params) {
  return runtime_->create_image(params);
}

void OpenglProgramImpl::dump_cache_data_to_disk() {
  const auto &mgr = get_cache_manager();
  mgr->clean_offline_cache(offline_cache::string_to_clean_cache_policy(
                               config->offline_cache_cleaning_policy),
                           config->offline_cache_max_size_of_files,
                           config->offline_cache_cleaning_factor);
  mgr->dump_with_merging();
}

void OpenglProgramImpl::finalize() {
  runtime_.reset();
  device_.reset();
  opengl::reset_opengl();
}

OpenglProgramImpl::~OpenglProgramImpl() {
  finalize();
}

const std::unique_ptr<gfx::CacheManager>
    &OpenglProgramImpl::get_cache_manager() {
  if (!cache_manager_) {
    TI_ASSERT(runtime_ && snode_tree_mgr_ && device_);
    using Mgr = gfx::CacheManager;
    Mgr::Params params;
    params.arch = config->arch;
    params.mode = config->offline_cache ? Mgr::MemAndDiskCache : Mgr::MemCache;
    params.cache_path = config->offline_cache_file_path;
    params.runtime = runtime_.get();
    params.compile_config = config;
    params.caps = device_->get_caps();
    params.compiled_structs = &snode_tree_mgr_->get_compiled_structs();
    cache_manager_ = std::make_unique<gfx::CacheManager>(std::move(params));
  }
  return cache_manager_;
}

}  // namespace taichi::lang
