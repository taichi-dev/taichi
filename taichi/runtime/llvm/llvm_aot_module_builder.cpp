#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"

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
  kcache.args.reserve(kernel->nested_parameters.size());
  for (const auto &p : kernel->nested_parameters)
    kcache.args.push_back(p);
  kcache.args_type = kernel->args_type;
  kcache.args_size = kernel->args_size;
  kcache.rets = kernel->rets;
  kcache.ret_size = kernel->ret_size;
  kcache.ret_type = kernel->ret_type;
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

LLVMCompiledKernel LlvmAotModuleBuilder::compile_kernel(Kernel *kernel) {
  const auto &ckd =
      compilation_manager_.load_or_compile(compile_config_, {}, *kernel);
  TI_ASSERT(arch_uses_llvm(ckd.arch()));
  return dynamic_cast<const LLVM::CompiledKernelData &>(ckd)
      .get_internal_data()
      .compiled_data.clone();
}

}  // namespace taichi::lang
