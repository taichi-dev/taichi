#ifdef TI_WITH_DX11

#include "taichi/runtime/program_impls/dx/dx_program.h"

#include "taichi/rhi/dx/dx_api.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
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

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  return register_params_to_executable(
      get_cache_manager()->load_or_compile(config, kernel), runtime_.get());
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = directx11::make_dx11_device();

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void Dx11ProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void Dx11ProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                             uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder() {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::dx11);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::dx11);
  }
}

DeviceAllocation Dx11ProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}

std::unique_ptr<aot::Kernel> Dx11ProgramImpl::make_aot_kernel(Kernel &kernel) {
  auto params = get_cache_manager()->load_or_compile(config, &kernel);
  return std::make_unique<gfx::KernelImpl>(runtime_.get(), std::move(params));
}

void Dx11ProgramImpl::dump_cache_data_to_disk() {
  const auto &mgr = get_cache_manager();
  mgr->clean_offline_cache(offline_cache::string_to_clean_cache_policy(
                               config->offline_cache_cleaning_policy),
                           config->offline_cache_max_size_of_files,
                           config->offline_cache_cleaning_factor);
  mgr->dump_with_merging();
}

const std::unique_ptr<gfx::CacheManager> &Dx11ProgramImpl::get_cache_manager() {
  if (!cache_manager_) {
    TI_ASSERT(runtime_ && snode_tree_mgr_ && device_);
    auto target_device = std::make_unique<aot::TargetDevice>(config->arch);
    device_->clone_caps(*target_device);
    using Mgr = gfx::CacheManager;
    Mgr::Params params;
    params.arch = config->arch;
    params.mode =
        offline_cache::enabled_wip_offline_cache(config->offline_cache)
            ? Mgr::MemAndDiskCache
            : Mgr::MemCache;
    params.cache_path = config->offline_cache_file_path;
    params.runtime = runtime_.get();
    params.target_device = std::move(target_device);
    params.compiled_structs = &snode_tree_mgr_->get_compiled_structs();
    cache_manager_ = std::make_unique<gfx::CacheManager>(std::move(params));
  }
  return cache_manager_;
}

}  // namespace lang
}  // namespace taichi

#endif
