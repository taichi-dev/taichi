#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include "taichi/runtime/llvm/launch_arg_info.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"

namespace taichi::lang {

void LlvmAotModuleBuilder::dump(const std::string &output_dir,
                                const std::string &filename) const {
  LlvmOfflineCacheFileWriter writer;
  writer.set_data(std::move(cache_));
  writer.dump(output_dir);

  dump_graph(output_dir);
}

void LlvmAotModuleBuilder::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled = compile_kernel(kernel);
  LlvmOfflineCache::KernelCacheData kcache;
  kcache.kernel_key = identifier;
  kcache.compiled_data = std::move(compiled);
  kcache.args = infer_launch_args(kernel);
  kcache.last_used_at = std::time(nullptr);
  kcache.created_at = std::time(nullptr);
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

void LlvmAotModuleBuilder::add_compiled_kernel(const std::string &identifier, aot::Kernel *kernel) {
  auto *kernel_impl = dynamic_cast<llvm_aot::KernelImpl *>(kernel);
  TI_ASSERT(kernel_impl);
  if (!kernel_impl->kernel_data_.created_at) {
    kernel_impl->kernel_data_.last_used_at = std::time(nullptr);
    kernel_impl->kernel_data_.created_at = std::time(nullptr);
  }
  const std::string &kernel_name = identifier;
  if (cache_.kernels.find(kernel_name) == cache_.kernels.end()) {
    cache_.kernels[kernel_name] = std::move(kernel_impl->kernel_data_);
  }
}

}  // namespace taichi::lang
