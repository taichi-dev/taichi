#include "taichi/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include "taichi/llvm/launch_arg_info.h"
#include "taichi/llvm/llvm_program.h"

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

void LlvmAotModuleBuilder::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // Field refers to a leaf node(Place SNode) in a SNodeTree.
  // It makes no sense to just serialize the leaf node or its corresponding
  // branch. Instead, the minimal unit we have to serialize is the entire
  // SNodeTree. Note that SNodeTree's uses snode_tree_id as its identifier,
  // rather than the field's name. (multiple fields may end up referring to the
  // same SNodeTree)

  // 1. Find snode_tree_id
  int snode_tree_id = rep_snode->get_snode_tree_id();

  // 2. Fetch Cache from the Program
  // Kernel compilation is not allowed until all the Fields are finalized,
  // so we finished SNodeTree compilation during AOTModuleBuilder construction.
  //
  // By the time "add_field_per_backend()" is called,
  // SNodeTrees should have already been finalized,
  // with compiled info stored in LlvmProgramImpl::cache_data_.
  TI_ASSERT(prog_ != nullptr);
  LlvmOfflineCache::FieldCacheData field_cache =
      prog_->get_cached_field(snode_tree_id);

  // 3. Update AOT Cache
  cache_.fields[snode_tree_id] = std::move(field_cache);
}

}  // namespace lang
}  // namespace taichi
