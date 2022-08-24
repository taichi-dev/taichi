#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"
#include "taichi/runtime/gfx/runtime.h"

namespace taichi {
namespace lang {
namespace gfx {

class OfflineCacheManager {
 public:
  using CompiledKernelData = gfx::GfxRuntime::RegisterParams;

  OfflineCacheManager(
      const std::string &cache_path,
      Arch arch,
      GfxRuntime *runtime,
      std::unique_ptr<aot::TargetDevice> &&target_device,
      const std::vector<spirv::CompiledSNodeStructs> &compiled_structs);
  aot::Kernel *load_cached_kernel(const std::string &key);
  FunctionType cache_kernel(const std::string &key, Kernel *kernel);
  void dump_with_mergeing() const;

 private:
  std::string path_;
  GfxRuntime *runtime_{nullptr};
  std::unique_ptr<AotModuleBuilder> caching_module_builder_{nullptr};
  std::unique_ptr<aot::Module> cached_module_{nullptr};
};

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
