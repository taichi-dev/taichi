#include "taichi/cache/gfx/cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/common/cleanup.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/util/lock.h"

namespace taichi {
namespace lang {
namespace gfx {

namespace {

constexpr char kMetadataFileLockName[] = "metadata.lock";
using CompiledKernelData = gfx::GfxRuntime::RegisterParams;

}  // namespace

CacheManager::CacheManager(Params &&init_params)
    : mode_(init_params.mode),
      runtime_(init_params.runtime),
      compiled_structs_(*init_params.compiled_structs) {
  TI_ASSERT(init_params.runtime);
  TI_ASSERT(init_params.target_device);

  path_ = offline_cache::get_cache_path_by_arch(init_params.cache_path,
                                                init_params.arch);

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
      params.runtime = runtime_;
      cached_module_ = gfx::make_aot_module(params, init_params.arch);
    }
  }

  caching_module_builder_ = std::make_unique<gfx::AotModuleBuilderImpl>(
      compiled_structs_, init_params.arch,
      std::move(init_params.target_device));
}

CompiledKernelData CacheManager::load_or_compile(CompileConfig *config,
                                                 Kernel *kernel) {
  if (kernel->is_evaluator) {
    spirv::lower(kernel);
    return gfx::run_codegen(kernel, runtime_->get_ti_device(),
                            compiled_structs_);
  }
  std::string kernel_key = make_kernel_key(config, kernel);
  if (mode_ > NotCache) {
    if (auto opt = this->try_load_cached_kernel(kernel, kernel_key)) {
      return *opt;
    }
  }
  return this->compile_and_cache_kernel(kernel_key, kernel);
}

void CacheManager::dump_with_merging() const {
  if (mode_ == MemAndDiskCache) {
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
}

std::optional<CompiledKernelData> CacheManager::try_load_cached_kernel(
    Kernel *kernel,
    const std::string &key) {
  if (mode_ == NotCache) {
    return std::nullopt;
  }
  // Find in memory-cache
  auto *cache_builder =
      static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  auto params_opt = cache_builder->try_get_kernel_register_params(key);
  if (params_opt.has_value()) {
    TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
             kernel->get_name(), key);
    kernel->mark_as_from_cache();
    // TODO: Support multiple SNodeTrees in AOT.
    params_opt->num_snode_trees = compiled_structs_.size();
    return params_opt;
  }
  // Find in disk-cache
  if (mode_ == MemAndDiskCache && cached_module_) {
    if (auto *aot_kernel = cached_module_->get_kernel(key)) {
      TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
               key);
      kernel->mark_as_from_cache();
      auto *aot_kernel_impl = static_cast<gfx::KernelImpl *>(aot_kernel);
      auto compiled = aot_kernel_impl->params();
      // TODO: Support multiple SNodeTrees in AOT.
      compiled.num_snode_trees = compiled_structs_.size();
      return compiled;
    }
  }
  return std::nullopt;
}

CompiledKernelData CacheManager::compile_and_cache_kernel(
    const std::string &key,
    Kernel *kernel) {
  TI_DEBUG_IF(mode_ == MemAndDiskCache, "Cache kernel '{}' (key='{}')",
              kernel->get_name(), key);
  auto *cache_builder =
      static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
  TI_ASSERT(cache_builder != nullptr);
  cache_builder->add(key, kernel);
  auto params_opt = cache_builder->try_get_kernel_register_params(key);
  TI_ASSERT(params_opt.has_value());
  // TODO: Support multiple SNodeTrees in AOT.
  params_opt->num_snode_trees = compiled_structs_.size();
  return *params_opt;
}

std::string CacheManager::make_kernel_key(CompileConfig *config,
                                          Kernel *kernel) const {
  if (mode_ < MemAndDiskCache) {
    return kernel->get_name();
  }
  auto key = kernel->get_cached_kernel_key();
  if (key.empty()) {
    key = get_hashed_offline_cache_key(config, kernel);
    kernel->set_kernel_key_for_cache(key);
  }
  return key;
}

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
