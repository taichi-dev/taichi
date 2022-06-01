#pragma once

#include "taichi/aot/module_loader.h"
#include "taichi/llvm/llvm_program.h"

namespace taichi {
namespace lang {

class LlvmAotModule : public aot::Module {
 public:
  explicit LlvmAotModule(const std::string &module_path,
                         LlvmProgramImpl *program)
      : program_(program),
        cache_reader_(LlvmOfflineCacheFileReader::make(module_path)) {
    TI_ASSERT(program_ != nullptr);
  }

  uint64_t version() const override {
    return 0;
  }

  size_t get_root_size() const override {
    return 0;
  }

 protected:
  virtual FunctionType convert_module_to_function(
      const std::string &name,
      LlvmOfflineCache::KernelCacheData &&loaded) = 0;

  LlvmOfflineCache::KernelCacheData load_kernel_from_cache(
      const std::string &name);

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override;

  LlvmProgramImpl *const program_{nullptr};
  std::unique_ptr<LlvmOfflineCacheFileReader> cache_reader_{nullptr};
};

}  // namespace lang
}  // namespace taichi
