#include "taichi/cache/metal/cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/metal/codegen_metal.h"
#include "taichi/common/version.h"
#include "taichi/program/kernel.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

namespace offline_cache {
template <>
struct CacheCleanerUtils<metal::CacheManager::Metadata> {
  using MetadataType = metal::CacheManager::Metadata;
  using KernelMetaData = MetadataType::KernelMetadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
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
    std::string fn = kernel_meta.kernel_key + "." + kMetalCacheFilenameExt;
    return {fn};
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    // Do nothing
    return;
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    return filename_extension(name) == kMetalCacheFilenameExt;
  }
};

}  // namespace offline_cache

namespace metal {

CacheManager::CacheManager(Params &&init_params)
    : config_(std::move(init_params)) {
  if (config_.mode == MemAndDiskCache) {
    const auto filepath = join_path(config_.cache_path, kMetadataFilename);
    const auto lock_path = join_path(config_.cache_path, kMetadataLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    } else {
      TI_WARN(
          "Lock {} failed. You can run 'ti cache clean -p {}' and try again.",
          lock_path, config_.cache_path);
    }
  }
}

CacheManager::CompiledKernelData CacheManager::load_or_compile(
    const CompileConfig *compile_config,
    Kernel *kernel) {
  if (kernel->is_evaluator || config_.mode == NotCache) {
    return compile_kernel(kernel);
  }
  TI_ASSERT(config_.mode > NotCache);
  const auto kernel_key = make_kernel_key(compile_config, kernel);
  if (auto opt = try_load_cached_kernel(kernel, kernel_key)) {
    return *opt;
  }
  return compile_and_cache_kernel(kernel_key, kernel);
}

void CacheManager::dump_with_merging() const {
  if (config_.mode == MemAndDiskCache && !caching_kernels_.empty()) {
    taichi::create_directories(config_.cache_path);
    const auto filepath = join_path(config_.cache_path, kMetadataFilename);
    const auto lock_path = join_path(config_.cache_path, kMetadataLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      Metadata data;
      data.version[0] = TI_VERSION_MAJOR;
      data.version[1] = TI_VERSION_MINOR;
      data.version[2] = TI_VERSION_PATCH;
      // Load old cached data
      load_metadata_with_checking(data, filepath);
      // Update the cached data
      for (const auto *e : updated_data_) {
        auto iter = data.kernels.find(e->kernel_key);
        if (iter != data.kernels.end()) {
          iter->second.last_used_at = e->last_used_at;
        }
      }
      // Add new data
      for (auto &[key, kernel] : caching_kernels_) {
        auto [iter, ok] = data.kernels.insert({key, kernel});
        if (ok) {
          data.size += iter->second.size;
          // Mangle kernel_name as kernel-key
          iter->second.compiled_kernel_data.kernel_name = key;
        }
      }
      // Dump
      for (const auto &[_, k] : data.kernels) {
        const auto code_filepath = join_path(
            config_.cache_path, fmt::format(kMetalCodeFormat, k.kernel_key));
        if (try_lock_with_file(code_filepath)) {
          std::ofstream fs{code_filepath};
          fs << k.compiled_kernel_data.source_code;
        }
      }
      write_to_binary_file(data, filepath);
    }
  }
}

void CacheManager::clean_offline_cache(offline_cache::CleanCachePolicy policy,
                                       int max_bytes,
                                       double cleaning_factor) const {
  if (config_.mode == MemAndDiskCache) {
    using CacheCleaner = offline_cache::CacheCleaner<Metadata>;
    offline_cache::CacheCleanerConfig params;
    params.path = config_.cache_path;
    params.policy = policy;
    params.cleaning_factor = cleaning_factor;
    params.max_size = max_bytes;
    params.metadata_filename = kMetadataFilename;
    params.debugging_metadata_filename = "";  // No debugging file
    params.metadata_lock_name = kMetadataLockName;
    CacheCleaner::run(params);
  }
}

CompiledKernelData CacheManager::compile_kernel(Kernel *kernel) const {
  kernel->lower();
  return run_codegen(config_.compiled_runtime_module_,
                     *config_.compiled_snode_trees_, kernel, nullptr);
}

std::string CacheManager::make_kernel_key(const CompileConfig *compile_config,
                                          Kernel *kernel) const {
  if (config_.mode < MemAndDiskCache) {
    return kernel->get_name();
  }
  auto key = kernel->get_cached_kernel_key();
  if (key.empty()) {
    key = get_hashed_offline_cache_key(compile_config, kernel);
    kernel->set_kernel_key_for_cache(key);
  }
  return key;
}

std::optional<CompiledKernelData> CacheManager::try_load_cached_kernel(
    Kernel *kernel,
    const std::string &key) {
  TI_ASSERT(config_.mode > NotCache);
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(key);
    if (iter != kernels.end()) {
      TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel->get_name(), key);
      kernel->mark_as_from_cache();
      return iter->second.compiled_kernel_data;
    }
  }
  // Find in disk-cache (cached_data_)
  if (config_.mode == MemAndDiskCache) {
    auto &kernels = cached_data_.kernels;
    auto iter = kernels.find(key);
    if (iter != kernels.end()) {
      auto &k = iter->second;
      TI_ASSERT(k.kernel_key == k.compiled_kernel_data.kernel_name);
      if (load_kernel_source_code(k)) {
        TI_DEBUG("Create kernel '{}' from cache (key='{}')", kernel->get_name(),
                 key);
        k.last_used_at = std::time(nullptr);
        updated_data_.push_back(&k);
        kernel->mark_as_from_cache();
        return k.compiled_kernel_data;
      }
    }
  }
  return std::nullopt;
}

CompiledKernelData CacheManager::compile_and_cache_kernel(
    const std::string &key,
    Kernel *kernel) {
  TI_DEBUG_IF(config_.mode == MemAndDiskCache, "Cache kernel '{}' (key='{}')",
              kernel->get_name(), key);
  OfflineCacheKernelMetadata k;
  k.kernel_key = key;
  k.created_at = k.last_used_at = std::time(nullptr);
  k.compiled_kernel_data = compile_kernel(kernel);
  k.size = k.compiled_kernel_data.source_code.size();
  const auto &kernel_data = (caching_kernels_[key] = std::move(k));
  return kernel_data.compiled_kernel_data;
}

bool CacheManager::load_kernel_source_code(
    OfflineCacheKernelMetadata &kernel_data) {
  auto &src = kernel_data.compiled_kernel_data.source_code;
  if (!src.empty()) {
    return true;
  }
  auto filepath =
      join_path(config_.cache_path,
                fmt::format(kMetalCodeFormat, kernel_data.kernel_key));
  std::ifstream f(filepath);
  if (!f.is_open()) {
    return false;
  }
  f.seekg(0, std::ios::end);
  const auto len = f.tellg();
  f.seekg(0, std::ios::beg);
  src.resize(len);
  f.read(&src[0], len);
  return !f.fail();
}

}  // namespace metal
}  // namespace taichi::lang
