#pragma once

#include "taichi/aot/module_loader.h"
#include "taichi/llvm/llvm_program.h"

namespace taichi {
namespace lang {

TI_DLL_EXPORT void finalize_aot_field(aot::Module *aot_module,
                                      aot::Field *aot_field,
                                      uint64 *result_buffer);

class LlvmAotModule : public aot::Module {
 public:
  explicit LlvmAotModule(const std::string &module_path,
                         LlvmProgramImpl *program)
      : program_(program),
        cache_reader_(LlvmOfflineCacheFileReader::make(module_path)) {
    TI_ASSERT(program_ != nullptr);
  }

  Arch arch() const override {
    return program_->config->arch;
  }

  uint64_t version() const override {
    return 0;
  }

  size_t get_root_size() const override {
    return 0;
  }

  LlvmProgramImpl *const get_program() {
    return program_;
  }

  void set_initialized_snode_tree(int snode_tree_id) {
    initialized_snode_tree_ids.insert(snode_tree_id);
  }

  bool is_snode_tree_initialized(int snode_tree_id) {
    return initialized_snode_tree_ids.count(snode_tree_id);
  }

 protected:
  virtual FunctionType convert_module_to_function(
      const std::string &name,
      LlvmOfflineCache::KernelCacheData &&loaded) = 0;

  LlvmOfflineCache::KernelCacheData load_kernel_from_cache(
      const std::string &name);

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override;

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override;

  LlvmProgramImpl *const program_{nullptr};
  std::unique_ptr<LlvmOfflineCacheFileReader> cache_reader_{nullptr};

  // To prevent repeated SNodeTree initialization
  std::unordered_set<int> initialized_snode_tree_ids;
};

}  // namespace lang
}  // namespace taichi
