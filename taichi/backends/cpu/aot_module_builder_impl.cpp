#include "taichi/backends/cpu/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/backends/cpu/codegen_cpu.h"

namespace taichi {
namespace lang {
namespace cpu {

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  LlvmOfflineCacheFileWriter writer;
  writer.set_data(std::move(cache_));
  writer.dump(output_dir);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto cgen = CodeGenCPU::make_codegen_llvm(kernel, /*ir=*/nullptr);
  auto compiled = cgen->run_compilation();
  LlvmOfflineCache::KernelCacheData kcache;
  kcache.kernel_key = identifier;
  kcache.module = compiled.llvm_module.get();
  kcache.owned_module = std::move(compiled.llvm_module);
  const auto &tasks = compiled.offloaded_tasks;
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

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
