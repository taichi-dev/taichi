#include "taichi/cache/gfx/cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/common/cleanup.h"
#include "taichi/common/version.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/util/lock.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

namespace {

constexpr char kMetadataFileLockName[] = "metadata.lock";
constexpr char kAotMetadataFilename[] = "metadata.tcb";
constexpr char kDebuggingAotMetadataFilename[] = "metadata.json";
constexpr char kGraphMetadataFilename[] = "graphs.tcb";
constexpr char kOfflineCacheMetadataFilename[] = "offline_cache_metadata.tcb";
using CompiledKernelData = gfx::GfxRuntime::RegisterParams;

inline gfx::CacheManager::Metadata::KernelMetadata make_kernel_metadata(
    const std::string &key,
    const gfx::GfxRuntime::RegisterParams &compiled) {
  std::size_t codes_size = 0;
  for (const auto &e : compiled.task_spirv_source_codes) {
    codes_size += e.size() * sizeof(*e.data());
  }

  gfx::CacheManager::Metadata::KernelMetadata res;
  res.kernel_key = key;
  res.size = codes_size;
  res.created_at = std::time(nullptr);
  res.last_used_at = std::time(nullptr);
  res.num_files = compiled.task_spirv_source_codes.size();
  return res;
}

}  // namespace

namespace offline_cache {

template <>
struct CacheCleanerUtils<gfx::CacheManager::Metadata> {
  using MetadataType = gfx::CacheManager::Metadata;
  using KernelMetaData = MetadataType::KernelMetadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    // Update AOT metadata
    gfx::TaichiAotData old_aot_data, new_aot_data;
    auto aot_metadata_path =
        taichi::join_path(config.path, kAotMetadataFilename);
    if (read_from_binary_file(old_aot_data, aot_metadata_path)) {
      const auto &kernels = data.kernels;
      for (auto &k : old_aot_data.kernels) {
        if (kernels.count(k.name)) {
          new_aot_data.kernels.push_back(std::move(k));
        }
      }
      write_to_binary_file(new_aot_data, aot_metadata_path);
    }
    write_to_binary_file(
        data, taichi::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    // Do nothing
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const KernelMetaData &kernel_meta) {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < kernel_meta.num_files; ++i) {
      result.push_back(kernel_meta.kernel_key + std::to_string(i) + "." +
                       kSpirvCacheFilenameExt);
    }
    return result;
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    taichi::remove(taichi::join_path(config.path, kAotMetadataFilename));
    taichi::remove(
        taichi::join_path(config.path, kDebuggingAotMetadataFilename));
    taichi::remove(taichi::join_path(config.path, kGraphMetadataFilename));
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    return filename_extension(name) == kSpirvCacheFilenameExt;
  }
};

}  // namespace offline_cache

namespace gfx {

CacheManager::CacheManager(Params &&init_params)
    : mode_(init_params.mode),
      runtime_(init_params.runtime),
      compile_config_(*init_params.compile_config),
      compiled_structs_(*init_params.compiled_structs) {
  TI_ASSERT(init_params.runtime);
  TI_ASSERT(init_params.compile_config);
  TI_ASSERT(init_params.compiled_structs);

  path_ = offline_cache::get_cache_path_by_arch(init_params.cache_path,
                                                init_params.arch);
  {  // Load cached module with checking
    using Error = offline_cache::LoadMetadataError;
    using offline_cache::load_metadata_with_checking;
    Metadata tmp;
    auto filepath = taichi::join_path(path_, kOfflineCacheMetadataFilename);
    if (load_metadata_with_checking(tmp, filepath) == Error::kNoError) {
      auto lock_path = taichi::join_path(path_, kMetadataFileLockName);
      auto exists =
          taichi::path_exists(taichi::join_path(path_, kAotMetadataFilename)) &&
          taichi::path_exists(taichi::join_path(path_, kGraphMetadataFilename));
      if (exists) {
        if (lock_with_file(lock_path)) {
          auto _ = make_cleanup([&lock_path]() {
            if (!unlock_with_file(lock_path)) {
              TI_WARN(
                  "Unlock {} failed. You can remove this .lock file manually "
                  "and try again.",
                  lock_path);
            }
          });
          gfx::AotModuleParams params;
          params.module_path = path_;
          params.runtime = runtime_;
          params.enable_lazy_loading = true;
          cached_module_ = gfx::make_aot_module(params, init_params.arch);
        } else {
          TI_WARN(
              "Lock {} failed. You can run 'ti cache clean -p {}' and try "
              "again.",
              lock_path, path_);
        }
      }
    }
  }

  caching_module_builder_ = std::make_unique<gfx::AotModuleBuilderImpl>(
      compiled_structs_, init_params.arch, compile_config_,
      std::move(init_params.caps));

  offline_cache_metadata_.version[0] = TI_VERSION_MAJOR;
  offline_cache_metadata_.version[1] = TI_VERSION_MINOR;
  offline_cache_metadata_.version[2] = TI_VERSION_PATCH;
}

CompiledKernelData CacheManager::load_or_compile(const CompileConfig &config,
                                                 Kernel *kernel) {
  if (kernel->is_evaluator) {
    spirv::lower(config, kernel);
    return gfx::run_codegen(kernel, runtime_->get_ti_device()->arch(),
                            runtime_->get_ti_device()->get_caps(),
                            compiled_structs_, config);
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
  if (mode_ == MemAndDiskCache && !offline_cache_metadata_.kernels.empty()) {
    taichi::create_directories(path_);
    auto *cache_builder =
        static_cast<gfx::AotModuleBuilderImpl *>(caching_module_builder_.get());
    cache_builder->mangle_aot_data();

    auto lock_path = taichi::join_path(path_, kMetadataFileLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_cleanup([&lock_path]() {
        if (!unlock_with_file(lock_path)) {
          TI_WARN(
              "Unlock {} failed. You can remove this .lock file manually and "
              "try again.",
              lock_path);
        }
      });

      // Update metadata.{tcb,json}
      cache_builder->merge_with_old_meta_data(path_);
      cache_builder->dump(path_, "");

      // Update offline_cache_metadata.tcb
      using offline_cache::load_metadata_with_checking;
      using Error = offline_cache::LoadMetadataError;
      Metadata old_data;
      const auto filename =
          taichi::join_path(path_, kOfflineCacheMetadataFilename);
      if (load_metadata_with_checking(old_data, filename) == Error::kNoError) {
        for (auto &[k, v] : offline_cache_metadata_.kernels) {
          auto iter = old_data.kernels.find(k);
          if (iter != old_data.kernels.end()) {  // Update
            iter->second.last_used_at = v.last_used_at;
          } else {  // Add new
            old_data.size += v.size;
            old_data.kernels[k] = std::move(v);
          }
        }
        write_to_binary_file(old_data, filename);
      } else {
        write_to_binary_file(offline_cache_metadata_, filename);
      }
    }
  }
}

void CacheManager::clean_offline_cache(offline_cache::CleanCachePolicy policy,
                                       int max_bytes,
                                       double cleaning_factor) const {
  if (mode_ == MemAndDiskCache) {
    using CacheCleaner = offline_cache::CacheCleaner<Metadata>;
    offline_cache::CacheCleanerConfig params;
    params.path = path_;
    params.policy = policy;
    params.cleaning_factor = cleaning_factor;
    params.max_size = max_bytes;
    params.metadata_filename = kOfflineCacheMetadataFilename;
    params.debugging_metadata_filename = "";  // No debugging file
    params.metadata_lock_name = kMetadataFileLockName;
    CacheCleaner::run(params);
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
    // TODO: Support multiple SNodeTrees in AOT.
    params_opt->num_snode_trees = compiled_structs_.size();
    return params_opt;
  }
  // Find in disk-cache
  if (mode_ == MemAndDiskCache && cached_module_) {
    if (auto *aot_kernel = cached_module_->get_kernel(key)) {
      TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
               key);
      auto *aot_kernel_impl = static_cast<gfx::KernelImpl *>(aot_kernel);
      auto compiled = aot_kernel_impl->params();
      // TODO: Support multiple SNodeTrees in AOT.
      compiled.num_snode_trees = compiled_structs_.size();
      auto kmetadata = make_kernel_metadata(key, compiled);
      offline_cache_metadata_.size += kmetadata.size;
      offline_cache_metadata_.kernels[key] = std::move(kmetadata);
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
  auto kmetadata = make_kernel_metadata(key, *params_opt);
  offline_cache_metadata_.size += kmetadata.size;
  offline_cache_metadata_.kernels[key] = std::move(kmetadata);
  return *params_opt;
}

std::string CacheManager::make_kernel_key(const CompileConfig &config,
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
}  // namespace taichi::lang
