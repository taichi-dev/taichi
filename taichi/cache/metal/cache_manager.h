#pragma once

#include "taichi/codegen/metal/struct_metal.h"
#include "taichi/common/serialization.h"
#include "taichi/program/compile_config.h"
#include "taichi/runtime/metal/kernel_utils.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {
namespace metal {

struct OfflineCacheKernelMetadata {
  std::string kernel_key;
  std::size_t size{0};          // byte
  std::time_t created_at{0};    // sec
  std::time_t last_used_at{0};  // sec
  CompiledKernelData compiled_kernel_data;

  TI_IO_DEF(kernel_key, size, created_at, last_used_at, compiled_kernel_data);
};

class CacheManager {
  static constexpr char kMetadataFilename[] = "metadata.tcb";
  static constexpr char kMetadataLockName[] = "metadata.lock";
  static constexpr char kMetalCodeFormat[] =
      "{}.metal";  // "{kernel-key}.metal"
  using CompiledKernelData = taichi::lang::metal::CompiledKernelData;
  using CachingData =
      std::unordered_map<std::string, OfflineCacheKernelMetadata>;

 public:
  using Metadata = offline_cache::Metadata<OfflineCacheKernelMetadata>;
  enum Mode { NotCache, MemCache, MemAndDiskCache };

  struct Params {
    Mode mode{MemCache};
    std::string cache_path;
    const CompiledRuntimeModule *compiled_runtime_module_{nullptr};
    const std::vector<CompiledStructs> *compiled_snode_trees_{nullptr};
  };

  explicit CacheManager(Params &&init_params);

  // Load from memory || Load from disk || (Compile && Cache the result in
  // memory)
  CompiledKernelData load_or_compile(const CompileConfig *compile_config,
                                     Kernel *kernel);

  // Dump the cached data in memory to disk
  void dump_with_merging() const;

  // Run offline cache cleaning
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor) const;

 private:
  CompiledKernelData compile_kernel(const CompileConfig *compile_config,
                                    Kernel *kernel) const;
  std::string make_kernel_key(const CompileConfig *compile_config,
                              Kernel *kernel) const;
  std::optional<CompiledKernelData> try_load_cached_kernel(
      Kernel *kernel,
      const std::string &key);
  CompiledKernelData compile_and_cache_kernel(
      const std::string &key,
      const CompileConfig *compile_config,
      Kernel *kernel);
  bool load_kernel_source_code(OfflineCacheKernelMetadata &kernel_data);

  Params config_;
  CachingData caching_kernels_;
  Metadata cached_data_;
  std::vector<Metadata::KernelMetadata *> updated_data_;
};

}  // namespace metal
}  // namespace taichi::lang
