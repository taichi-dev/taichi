#include "taichi/runtime/gfx/offline_cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/common/cleanup.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/util/lock.h"

namespace taichi {
namespace lang {
namespace gfx {

namespace {

constexpr char kMetadataFileLockName[] = "metadata.lock";

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
    const std::vector<spirv::CompiledSNodeStructs> &compiled_structs)
    : runtime_(runtime) {
  path_ = offline_cache::get_cache_path_by_arch(cache_path, arch);

  if (taichi::path_exists(taichi::join_path(path_, "metadata.tcb")) &&
      taichi::path_exists(taichi::join_path(path_, "graphs.tcb"))) {
    auto lock_path = taichi::join_path(path_, kMetadataFileLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_cleanup([&lock_path]() {
        if (!unlock_with_file(lock_path)) {
          TI_WARN("Unlock {} failed", lock_path);
        }
      });
      gfx::AotModuleParams params;
      params.module_path = path_;
      params.runtime = runtime;
      cached_module_ = gfx::make_aot_module(params, arch);
    }
  }

  caching_module_builder_ = std::make_unique<gfx::AotModuleBuilderImpl>(
      compiled_structs, arch, std::move(target_device));
}

FunctionType OfflineCacheManager::load_or_compile(CompileConfig *config,
                                                  Kernel *kernel) {
  auto kernel_key = get_hashed_offline_cache_key(config, kernel);
  kernel->set_kernel_key_for_cache(kernel_key);
  if (auto *cached_kernel = this->load_cached_kernel(kernel_key)) {
    // Load from cache
    TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
             kernel_key);
    kernel->set_from_offline_cache();
    return
        [cached_kernel](RuntimeContext &ctx) { cached_kernel->launch(&ctx); };
  }

  // Compile & Cache it
  TI_DEBUG("Cache kernel '{}' (key='{}')", kernel->get_name(), kernel_key);
  return this->cache_kernel(kernel_key, kernel);
}

aot::Kernel *OfflineCacheManager::load_cached_kernel(const std::string &key) {
  return cached_module_ ? cached_module_->get_kernel(key) : nullptr;
}

FunctionType OfflineCacheManager::cache_kernel(const std::string &key,
                                               Kernel *kernel) {
  auto *cache_builder =
      static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  TI_ASSERT(cache_builder != nullptr);
  cache_builder->add(key, kernel);
  auto params_opt = cache_builder->try_get_kernel_register_params(key);
  TI_ASSERT(params_opt.has_value());
  return register_params_to_executable(std::move(*params_opt), runtime_);
}

void OfflineCacheManager::dump_with_merging() const {
  taichi::create_directories(path_);
  auto *cache_builder =
      static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  cache_builder->mangle_aot_data();

  auto lock_path = taichi::join_path(path_, kMetadataFileLockName);
  if (lock_with_file(lock_path)) {
    auto _ = make_cleanup([&lock_path]() {
      if (!unlock_with_file(lock_path)) {
        TI_WARN("Unlock {} failed", lock_path);
      }
    });
    cache_builder->merge_with_old_meta_data(path_);
    cache_builder->dump(path_, "");
  }
}

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
