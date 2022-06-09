#include "taichi/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include "taichi/llvm/launch_arg_info.h"

namespace taichi {
namespace lang {

void LlvmAotModuleBuilder::dump(const std::string &output_dir,
                                const std::string &filename) const {
  LlvmOfflineCacheFileWriter writer;
  writer.set_data(std::move(cache_));
  writer.dump(output_dir);
}

void LlvmAotModuleBuilder::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled = compile_kernel(kernel);
  LlvmOfflineCache::KernelCacheData kcache;
  kcache.kernel_key = identifier;
  kcache.module = compiled.llvm_module.get();
  kcache.owned_module = std::move(compiled.llvm_module);
  const auto &tasks = compiled.offloaded_tasks;
  kcache.args = infer_launch_args(kernel);
  kcache.offloaded_task_list.resize(tasks.size());
  std::transform(tasks.begin(), tasks.end(), kcache.offloaded_task_list.begin(),
                 [](const auto &t) -> LlvmOfflineCache::OffloadedTaskCacheData {
                   LlvmOfflineCache::OffloadedTaskCacheData res;
                   res.name = t.name;
                   res.block_dim = t.block_dim;
                   res.grid_dim = t.grid_dim;
                   return res;
                 });
  cache_.kernels[identifier] = std::move(kcache);
}

}  // namespace lang
}  // namespace taichi
