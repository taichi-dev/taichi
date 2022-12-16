#include "taichi/cache/kernel_compilation_manager.h"

#include "taichi/rhi/arch.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/spirv/cxompiled_kernel_data.h"

namespace taichi::lang {

KernelCompilationManager::KernelCompilationManager(Config config)
    : config_(std::move(config)) {
  TI_DEBUG("Create KernelCompilationManager with offline_cache_file_path = {}",
           config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
  auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);
  if (lock_with_file(lock_path)) {
    auto _ = make_unlocker(lock_path);
    offline_cache::load_metadata_with_checking(cached_data_, filepath);
  }
}

const CompiledKernelData &KernelCompilationManager::load_or_compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def) {
  auto cache_mode = compile_config.offline_cache ? MemAndDiskCache : MemCache;
  const auto gukk = make_gukk(compile_config, kernel_def);
  auto cached_kernel =
      try_load_cached_kernel(kernel_def, gukk, compile_config.arch, cache_mode);
  return cached_kernel ? *cached_kernel
                       : compile_and_cache_kernel(gukk, compile_config,
                                                  device_caps, kernel_def);
}

void KernelCompilationManager::dump_with_merging() {
  if (!caching_kernels_.empty()) {
    taichi::create_directories(config_.offline_cache_path);
    auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
    auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      OfflineCacheData data;
      data.version[0] = TI_VERSION_MAJOR;
      data.version[1] = TI_VERSION_MINOR;
      data.version[2] = TI_VERSION_PATCH;
      // Load old cached data
      offline_cache::load_metadata_with_checking(data, filepath);
      // Update the cached data
      for (const auto *e : updated_data_) {
        auto iter = data.kernels.find(e->kernel_key);
        if (iter != data.kernels.end()) {
          iter->second.last_used_at = e->last_used_at;
        }
      }
      // Add new data
      for (auto &[gukk, kernel] : caching_kernels_) {
        auto [iter, ok] = data.kernels.insert({gukk, std::move(kernel)});
        // data.kernels.insert()
        if (ok) {
          data.size += iter->second.size;
          // TODO: Mangle iter->second.compiled_kernel_data->mangle_names(gukk);
        }
      }
      // Dump
      for (const auto &[_, k] : data.kernels) {
        auto cache_filename = join_path(
            config_.offline_cache_path,
            fmt::format(kCacheFilenameFormat, k.kernel_key, arch_name(k.arch)));
        if (k.compiled_kernel_data && try_lock_with_file(cache_filename)) {
          std::ofstream fs{cache_filename, std::ios::out | std::ios::binary};
          TI_ASSERT(fs.is_open());
          k.compiled_kernel_data->dump(fs);
        }
      }
      write_to_binary_file(data, filepath);
    }
  }
}

void KernelCompilationManager::clean_offline_cache(
    offline_cache::CleanCachePolicy policy,
    int max_bytes,
    double cleaning_factor) const {
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def) const {
  auto &compiler = *config_.kernel_compiler;
  auto ir = compiler.compile(compile_config, kernel_def);
  auto ckd = compiler.compile(compile_config, device_caps, kernel_def, *ir);
  return ckd;
}

std::string KernelCompilationManager::make_gukk(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  auto gukk = kernel_def.get_cached_kernel_key();
  if (gukk.empty()) {
    gukk = get_hashed_offline_cache_key(&compile_config, &kernel_def);
    kernel_def.set_kernel_key_for_cache(gukk);
  }
  return gukk;
}

const CompiledKernelData *KernelCompilationManager::try_load_cached_kernel(
    const Kernel &kernel_def,
    const std::string &gukk,
    Arch arch,
    CacheMode cache_mode) {
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(gukk);
    if (iter != kernels.end()) {
      TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel_def.get_name(), gukk);
      return iter->second.compiled_kernel_data.get();
    }
  }
  // Find in disk-cache (cached_data_)
  if (cache_mode == MemAndDiskCache) {
    auto &kernels = cached_data_.kernels;
    auto iter = kernels.find(gukk);
    if (iter != kernels.end()) {
      auto &k = iter->second;
      if (auto loaded = load_cache_file(gukk, arch)) {
        TI_DEBUG("Create kernel '{}' from cache (key='{}')",
                 kernel_def.get_name(), gukk);
        k.last_used_at = std::time(nullptr);
        k.compiled_kernel_data = std::move(loaded);
        updated_data_.push_back(&k);
        return k.compiled_kernel_data.get();
      }
    }
  }
  return nullptr;
}

const CompiledKernelData &KernelCompilationManager::compile_and_cache_kernel(
    const std::string &gukk,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def) {
  auto cache_mode = compile_config.offline_cache ? MemAndDiskCache : MemCache;
  TI_DEBUG_IF(cache_mode == MemAndDiskCache, "Cache kernel '{}' (key='{}')",
              kernel_def.get_name(), gukk);
  TI_ASSERT(caching_kernels_.find(gukk) == caching_kernels_.end());
  OfflineCacheKernelData k;
  k.arch = compile_config.arch;
  k.kernel_key = gukk;
  k.created_at = k.last_used_at = std::time(nullptr);
  k.compiled_kernel_data =
      compile_kernel(compile_config, device_caps, kernel_def);
  k.size = k.compiled_kernel_data->size();
  const auto &kernel_data = (caching_kernels_[gukk] = std::move(k));
  return *kernel_data.compiled_kernel_data;
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::load_cache_file(
    const std::string &gukk,
    Arch arch) {
  std::unique_ptr<CompiledKernelData> res;

  // FIXME: NO if BUT factory function
  if (arch_uses_spirv(arch)) {
    res = std::make_unique<spirv::CompiledKernelData>();
  } else {
    TI_NOT_IMPLEMENTED;
  }

  auto basename = fmt::format(kCacheFilenameFormat, gukk, arch_name(arch));
  auto filename = join_path(config_.offline_cache_path, basename);
  if (std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ifs.is_open()) {
    using Err = CompiledKernelData::Err;
    return res->load(ifs) == Err::kNoError ? std::move(res) : nullptr;
  }
  return nullptr;
}

}  // namespace taichi::lang
