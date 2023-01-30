#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {
namespace gfx {

struct OfflineCacheKernelMetadata {
  std::string kernel_key;
  std::size_t size{0};          // byte
  std::time_t created_at{0};    // sec
  std::time_t last_used_at{0};  // sec
  std::size_t num_files{0};

  TI_IO_DEF(kernel_key, size, created_at, last_used_at, num_files);
};

class CacheManager {
  using CompiledKernelData = gfx::GfxRuntime::RegisterParams;

 public:
  using Metadata = offline_cache::Metadata<OfflineCacheKernelMetadata>;
  enum Mode { NotCache, MemCache, MemAndDiskCache };

  struct Params {
    Arch arch;
    Mode mode{MemCache};
    std::string cache_path;
    GfxRuntime *runtime{nullptr};
    const CompileConfig *compile_config{nullptr};
    DeviceCapabilityConfig caps{};
    const std::vector<spirv::CompiledSNodeStructs> *compiled_structs;
  };

  explicit CacheManager(Params &&init_params);

  CompiledKernelData load_or_compile(const CompileConfig &config,
                                     Kernel *kernel);
  void dump_with_merging() const;
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor) const;

 private:
  std::optional<CompiledKernelData> try_load_cached_kernel(
      Kernel *kernel,
      const std::string &key);
  CompiledKernelData compile_and_cache_kernel(const std::string &key,
                                              Kernel *kernel);
  std::string make_kernel_key(const CompileConfig &config,
                              Kernel *kernel) const;

  Mode mode_{MemCache};
  std::string path_;
  GfxRuntime *runtime_{nullptr};
  const CompileConfig &compile_config_;
  const std::vector<spirv::CompiledSNodeStructs> &compiled_structs_;
  Metadata offline_cache_metadata_;
  std::unique_ptr<AotModuleBuilder> caching_module_builder_{nullptr};
  std::unique_ptr<aot::Module> cached_module_{nullptr};
};

}  // namespace gfx
}  // namespace taichi::lang
