#include "taichi/runtime/gfx/offline_cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
namespace gfx {

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

OfflineCacheManager::OfflineCacheManager(
                      const std::string &cache_path,
                      Arch arch,
                      GfxRuntime *runtime,
                      std::unique_ptr<aot::TargetDevice> &&target_device,
                      const std::vector<spirv::CompiledSNodeStructs> &compiled_structs) : runtime_(runtime) {
  path_ = offline_cache::get_cache_path_by_arch(cache_path, arch);

  if (taichi::path_exists(taichi::join_path(path_, "metadata.tcb")) &&
      taichi::path_exists(taichi::join_path(path_, "graphs.tcb"))) {
    gfx::AotModuleParams params;
    params.module_path = path_;
    params.runtime = runtime;
    cached_module_ = gfx::make_aot_module(params, arch);
  }

  caching_module_builder_ = std::make_unique<gfx::AotModuleBuilderImpl>(compiled_structs, arch, std::move(target_device));
}

aot::Kernel *OfflineCacheManager::load_cached_kernel(const std::string &key) {
  return cached_module_ ? cached_module_->get_kernel(key) : nullptr;
}

FunctionType OfflineCacheManager::cache_kernel(const std::string &key, Kernel *kernel) {
  auto *cache_builder = static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  TI_ASSERT(cache_builder != nullptr);
  cache_builder->add(key, kernel);
  auto params_opt = cache_builder->try_get_kernel_register_params(key);
  TI_ASSERT(params_opt.has_value());
  return register_params_to_executable(std::move(*params_opt), runtime_);
}

void OfflineCacheManager::dump_with_mergeing() const {
  taichi::create_directories(path_);
  auto *cache_builder = static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  cache_builder->mangle_aot_data();
  cache_builder->merge_with_old_meta_data(path_);
  cache_builder->dump(path_, "");
}

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
