#include "taichi/cache/metal/cache_manager.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/metal/codegen_metal.h"
#include "taichi/common/version.h"
#include "taichi/program/kernel.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {
namespace metal {

CacheManager::CacheManager(Params &&init_params)
    : config_(std::move(init_params)) {
  if (config_.mode == MemAndDiskCache) {
    const auto filepath = join_path(config_.cache_path, kMetadataFilename);
    const auto lock_path = join_path(config_.cache_path, kMetadataLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
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
