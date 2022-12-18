#pragma once

#include <ctime>
#include <string>
#include <memory>
#include <unordered_map>

#include "taichi/util/offline_cache.h"
#include "taichi/common/serialization.h"
#include "taichi/codegen/kernel_compiler.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {

struct CacheData {
  enum CacheMode { MemCache, MemAndDiskCache };
  using Version = std::uint16_t[3];

  struct KernelData {
    Arch arch;
    std::string kernel_key;
    std::size_t size{0};          // byte
    std::time_t created_at{0};    // sec
    std::time_t last_used_at{0};  // sec

    std::unique_ptr<lang::CompiledKernelData> compiled_kernel_data;

    // Dump the kernel to disk if `cache_mode` == `MemAndDiskCache`
    CacheMode cache_mode{MemCache};

    TI_IO_DEF(arch, kernel_key, size, created_at, last_used_at);
  };

  Version version{};
  std::size_t size{0};
  std::unordered_map<std::string, KernelData> kernels;

  // NOTE: The "version" must be the first field to be serialized
  TI_IO_DEF(version, size, kernels);
};

class KernelCompilationManager final {
  static constexpr char kMetadataFilename[] = "ticache.tcb";
  static constexpr char kCacheFilenameFormat[] = "{}-{}.tic";
  static constexpr char kMetadataLockName[] = "ticache.lock";

  using KernelCacheData = CacheData::KernelData;
  using CachingKernels = std::unordered_map<std::string, KernelCacheData>;

 public:
  struct Config {
    std::string offline_cache_path;
    std::unique_ptr<KernelCompiler> kernel_compiler;
  };

  explicit KernelCompilationManager(Config init_params);

  // Load from memory || Load from disk || (Compile && Cache the result in
  // memory)
  const CompiledKernelData &load_or_compile(const CompileConfig &compile_config,
                                            const DeviceCapabilityConfig &caps,
                                            const Kernel &kernel_def);

  // Dump the cached data in memory to disk
  void dump_with_merging();

  // Run offline cache cleaning
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor) const;

 private:
  std::unique_ptr<CompiledKernelData> compile_kernel(
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &device_caps,
      const Kernel &kernel_def) const;

  std::string make_gukk(const CompileConfig &compile_config,
                        const Kernel &kernel_def) const;

  const CompiledKernelData *try_load_cached_kernel(
      const Kernel &kernel_def,
      const std::string &gukk,
      Arch arch,
      CacheData::CacheMode cache_mode);

  const CompiledKernelData &compile_and_cache_kernel(
      const std::string &gukk,
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &device_caps,
      const Kernel &kernel_def);

  std::unique_ptr<CompiledKernelData> load_cache_file(const std::string &gukk,
                                                      Arch arch);

  Config config_;
  CachingKernels caching_kernels_;
  CacheData cached_data_;
  std::vector<KernelCacheData *> updated_data_;
};

}  // namespace taichi::lang
