#include "taichi/cache/metal/cache_manager.h"
#include "taichi/common/serialization.h"
#include "taichi/common/version.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {
namespace metal {

CacheManager::CacheManager(Params &&init_params) : config_(std::move(init_params)) {
  if (config_.mode == MemAndDiskCache) {
    const auto filepath = join_path(config_.cache_path, kMetadataFilename);
    const auto lock_path = join_path(config_.cache_path, kMetadataLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    }
  }
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
      for (const auto &[key, kernel] : caching_kernels_) {
        auto [iter, ok] = data.kernels.insert({key, std::move(kernel)});
        if (ok) {
          data.size += iter->second.size;
        }
      }
      // Dump
      for (const auto &[_, k] : data.kernels) {
        const auto code_filepath = join_path(config_.cache_path, fmt::format(kMetalCodeFormat, k.kernel_key));
        if (try_lock_with_file(code_filepath)) {
          std::ofstream fs{code_filepath};
          fs << k.compiled_kernel_data.source_code;
        }
      }
      write_to_binary_file(data, filepath);
    }
  }
}

}  // namespace metal
}  // namespace taichi::lang
