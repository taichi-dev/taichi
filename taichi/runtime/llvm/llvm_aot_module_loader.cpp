#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/llvm/aot_graph_data.h"

namespace taichi {
namespace lang {

LlvmOfflineCache::KernelCacheData LlvmAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  auto *tlctx = executor_->get_llvm_context(executor_->get_config()->arch);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = cache_reader_->get_kernel_cache(loaded, name,
                                            *tlctx->get_this_thread_context());
  TI_ERROR_IF(!ok, "Failed to load kernel={}", name);
  return loaded;
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel(
    const std::string &name) {
  auto fn = convert_module_to_function(name, load_kernel_from_cache(name));
  return std::make_unique<llvm_aot::KernelImpl>(
      fn, name, LlvmOfflineCache::KernelCacheData());
}

std::unique_ptr<aot::Field> LlvmAotModule::make_new_field(
    const std::string &name) {
  // Check if "name" represents snode_tree_id.
  // Avoid using std::atoi due to its poor error handling.
  char *end;
  int snode_tree_id = static_cast<int>(strtol(name.c_str(), &end, 10 /*base*/));

  TI_ASSERT(end != name.c_str());
  TI_ASSERT(*end == '\0');

  // Load FieldCache
  LlvmOfflineCache::FieldCacheData loaded;
  auto ok = cache_reader_->get_field_cache(loaded, snode_tree_id);
  TI_ERROR_IF(!ok, "Failed to load field with id={}", snode_tree_id);

  return std::make_unique<llvm_aot::FieldImpl>(std::move(loaded));
}

std::unique_ptr<aot::CompiledGraph> LlvmAotModule::get_graph(
    const std::string &name) {
  auto it = graphs_.find(name);
  if (it == graphs_.end()) {
    TI_DEBUG("Cannot find graph {}", name);
    return nullptr;
  }

  std::vector<aot::CompiledDispatch> dispatches;
  for (auto &dispatch : it->second.dispatches) {
    dispatches.push_back({dispatch.kernel_name, dispatch.symbolic_args,
                          get_kernel(dispatch.kernel_name)});
  }

  aot::CompiledGraph graph = aot::CompiledGraph({dispatches});
  executor_->prepare_runtime_context(&graph.ctx_);

  return std::make_unique<aot::CompiledGraph>(std::move(graph));
}

void allocate_aot_snode_tree_type(aot::Module *aot_module,
                                  aot::Field *aot_field,
                                  uint64 *result_buffer) {
  auto *llvm_aot_module = dynamic_cast<LlvmAotModule *>(aot_module);
  auto *aot_field_impl = dynamic_cast<llvm_aot::FieldImpl *>(aot_field);

  TI_ASSERT(llvm_aot_module != nullptr);
  TI_ASSERT(aot_field_impl != nullptr);

  auto *runtime_executor = llvm_aot_module->get_runtime_executor();
  const auto &field_cache = aot_field_impl->get_snode_tree_cache();

  int snode_tree_id = field_cache.tree_id;
  if (!llvm_aot_module->is_snode_tree_initialized(snode_tree_id)) {
    runtime_executor->initialize_llvm_runtime_snodes(field_cache,
                                                     result_buffer);
    llvm_aot_module->set_initialized_snode_tree(snode_tree_id);
  }
}

}  // namespace lang
}  // namespace taichi
