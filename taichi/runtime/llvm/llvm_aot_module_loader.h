#pragma once

#include "taichi/aot/module_loader.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi {
namespace lang {

TI_DLL_EXPORT void finalize_aot_field(aot::Module *aot_module,
                                      aot::Field *aot_field,
                                      uint64 *result_buffer);

class LlvmAotModule : public aot::Module {
 public:
  explicit LlvmAotModule(const std::string &module_path,
                         LlvmRuntimeExecutor *executor)
      : executor_(executor),
        cache_reader_(LlvmOfflineCacheFileReader::make(module_path)) {
    TI_ASSERT(executor_ != nullptr);

    const std::string graph_path = fmt::format("{}/graphs.tcb", module_path);
    read_from_binary_file(graphs_, graph_path);
  }

  Arch arch() const override {
    return executor_->get_config()->arch;
  }

  uint64_t version() const override {
    return 0;
  }

  size_t get_root_size() const override {
    return 0;
  }

  LlvmRuntimeExecutor *const get_runtime_executor() {
    return executor_;
  }

  void set_initialized_snode_tree(int snode_tree_id) {
    initialized_snode_tree_ids.insert(snode_tree_id);
  }

  bool is_snode_tree_initialized(int snode_tree_id) {
    return initialized_snode_tree_ids.count(snode_tree_id);
  }

  std::unique_ptr<aot::CompiledGraph> get_graph(std::string name) override;

 protected:
  virtual FunctionType convert_module_to_function(
      const std::string &name,
      LlvmOfflineCache::KernelCacheData &&loaded) = 0;

  LlvmOfflineCache::KernelCacheData load_kernel_from_cache(
      const std::string &name);

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override;

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override;

  LlvmRuntimeExecutor *const executor_{nullptr};
  std::unique_ptr<LlvmOfflineCacheFileReader> cache_reader_{nullptr};

  // To prevent repeated SNodeTree initialization
  std::unordered_set<int> initialized_snode_tree_ids;
};

}  // namespace lang
}  // namespace taichi
