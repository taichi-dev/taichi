#include "taichi/com_mgr/kernel_compilation_manager.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {

KernelCompilationManager::KernelCompilationManager(Config config)
    : config_(std::move(config)) {
  using Error = offline_cache::LoadMetadataError;

  TI_DEBUG("Create KernelCompilationManager with offline_cache_file_path = {}",
           config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
  auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);
  if (path_exists(filepath)) {
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    } else {
      TI_WARN(
          "Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
          lock_path, config_.offline_cache_path);
    }
  }
}

const CompiledKernelData &KernelCompilationManager::load_or_compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto cache_mode = compile_config.offline_cache ? CacheData::MemAndDiskCache
                                                 : CacheData::MemCache;
  const auto kernel_key = make_kernel_key(compile_config, caps, kernel_def);
  auto cached_kernel = try_load_cached_kernel(kernel_def, kernel_key,
                                              compile_config.arch, cache_mode);
  return cached_kernel ? *cached_kernel
                       : compile_and_cache_kernel(kernel_key, compile_config,
                                                  caps, kernel_def);
}

void KernelCompilationManager::dump() {
  if (caching_kernels_.empty()) {
    return;
  }

  taichi::create_directories(config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
  auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);

  if (!lock_with_file(lock_path)) {
    TI_WARN("Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
            lock_path, config_.offline_cache_path);
  }

  auto _ = make_unlocker(lock_path);
  CacheData data;
  data.version[0] = TI_VERSION_MAJOR;
  data.version[1] = TI_VERSION_MINOR;
  data.version[2] = TI_VERSION_PATCH;
  auto &kernels = data.kernels;
  // Load old cached data
  offline_cache::load_metadata_with_checking(data, filepath);
  // Update the cached data
  for (const auto *e : updated_data_) {
    auto iter = kernels.find(e->kernel_key);
    if (iter != kernels.end()) {
      iter->second.last_used_at = e->last_used_at;
    }
  }
  // Add new data
  for (auto &[kernel_key, kernel] : caching_kernels_) {
    if (kernel.cache_mode == CacheData::MemAndDiskCache) {
      auto [iter, ok] = kernels.insert({kernel_key, std::move(kernel)});
      data.size += ok ? iter->second.size : 0;
    }
  }
  // Dump cached CompiledKernelData to disk
  for (const auto &[_, k] : kernels) {
    if (k.compiled_kernel_data) {
      const auto arch = k.compiled_kernel_data->arch();
      auto cache_filename = make_filename(k.kernel_key, arch);
      if (try_lock_with_file(cache_filename)) {
        std::ofstream fs{cache_filename, std::ios::out | std::ios::binary};
        TI_ASSERT(fs.is_open());
        k.compiled_kernel_data->dump(fs);
      }
    }
  }
  // Dump offline cache metadata
  if (!kernels.empty()) {
    write_to_binary_file(data, filepath);
  }
}

void KernelCompilationManager::clean_offline_cache(
    offline_cache::CleanCachePolicy policy,
    int max_bytes,
    double cleaning_factor) const {
  // TODO(PGZXB): Impl
}

std::string KernelCompilationManager::make_filename(
    const std::string &kernel_key,
    Arch arch) const {
  return join_path(
      config_.offline_cache_path,
      fmt::format(kCacheFilenameFormat, kernel_key, arch_name(arch)));
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto &compiler = *config_.kernel_compiler;
  auto ir = compiler.compile(compile_config, kernel_def);
  auto ckd = compiler.compile(compile_config, caps, kernel_def, *ir);
  return ckd;
}

std::string KernelCompilationManager::make_kernel_key(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto kernel_key = kernel_def.get_cached_kernel_key();
  if (kernel_key.empty()) {
    kernel_key = get_hashed_offline_cache_key(compile_config, caps,
                                              (Kernel *)&kernel_def);
    kernel_def.set_kernel_key_for_cache(kernel_key);
  }
  return kernel_key;
}

const CompiledKernelData *KernelCompilationManager::try_load_cached_kernel(
    const Kernel &kernel_def,
    const std::string &kernel_key,
    Arch arch,
    CacheData::CacheMode cache_mode) {
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel_def.get_name(), kernel_key);
      return iter->second.compiled_kernel_data.get();
    }
  }
  // Find in disk-cache (cached_data_)
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto &kernels = cached_data_.kernels;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      auto &k = iter->second;
      if (auto loaded = load_ckd(kernel_key, arch)) {
        TI_DEBUG("Create kernel '{}' from cache (key='{}')",
                 kernel_def.get_name(), kernel_key);
        TI_ASSERT(loaded->arch() == arch);
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
    const std::string &kernel_key,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto cache_mode = compile_config.offline_cache ? CacheData::MemAndDiskCache
                                                 : CacheData::MemCache;
  TI_DEBUG_IF(cache_mode == CacheData::MemAndDiskCache,
              "Cache kernel '{}' (key='{}')", kernel_def.get_name(),
              kernel_key);
  TI_ASSERT(caching_kernels_.find(kernel_key) == caching_kernels_.end());
  KernelCacheData k;
  k.kernel_key = kernel_key;
  k.created_at = k.last_used_at = std::time(nullptr);
  k.compiled_kernel_data = compile_kernel(compile_config, caps, kernel_def);
  k.size = k.compiled_kernel_data->size();
  k.cache_mode = cache_mode;
  const auto &kernel_data = (caching_kernels_[kernel_key] = std::move(k));
  return *kernel_data.compiled_kernel_data;
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::load_ckd(
    const std::string &kernel_key,
    Arch arch) {
  const auto filename = make_filename(kernel_key, arch);
  if (std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ifs.is_open()) {
    CompiledKernelData::Err err;
    auto ckd = CompiledKernelData::load(ifs, &err);
    if (err != CompiledKernelData::Err::kNoError) {
      TI_DEBUG("Load cached CompiledKernelData file failed: {}",
               CompiledKernelData::get_err_msg(err));
      return nullptr;
    }
    return ckd;
  }
  return nullptr;
}

}  // namespace taichi::lang
