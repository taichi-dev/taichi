#include <unordered_set>

#include "metal_program.h"
#include "taichi/backends/metal/codegen_metal.h"
#include "taichi/backends/metal/struct_metal.h"

namespace taichi {
namespace lang {
namespace {

std::unordered_set<const SNode *> find_all_dense_snodes(
    const metal::SNodeDescriptorsMap &snodes_map) {
  std::unordered_set<const SNode *> res;
  for (const auto &[_, desc] : snodes_map) {
    const auto *sn = desc.snode;
    if (sn->type == SNodeType::dense) {
      res.insert(sn);
    }
  }
  return res;
}

bool all_fields_are_dense(
    const std::unordered_set<const SNode *> &placed_snodes) {
  for (const auto *sn : placed_snodes) {
    for (const auto &ch : sn->ch) {
      if (ch->type != SNodeType::place) {
        return false;
      }
    }
    const auto *parent = sn->parent;
    if (!parent) {
      return false;
    }
    if (parent->type != SNodeType::root) {
      return false;
    }
  }
  return true;
}

}  // namespace

MetalProgramImpl::MetalProgramImpl(CompileConfig &config_)
    : ProgramImpl(config_) {
}

FunctionType MetalProgramImpl::compile(Kernel *kernel,
                                       OffloadedStmt *offloaded) {
  if (!kernel->lowered()) {
    kernel->lower();
  }
  return metal::compile_to_metal_executable(kernel, metal_kernel_mgr_.get(),
                                            &(compiled_runtime_module_.value()),
                                            compiled_snode_trees_, offloaded);
}

std::size_t MetalProgramImpl::get_snode_num_dynamically_allocated(SNode *snode,
                                                                  uint64 *) {
  // TODO: result_buffer is not used here since it's saved in params and already
  // available in metal_kernel_mgr
  return metal_kernel_mgr_->get_snode_num_dynamically_allocated(snode);
}

void MetalProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                           KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  TI_ASSERT(*result_buffer_ptr == nullptr);
  TI_ASSERT(metal_kernel_mgr_ == nullptr);
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  compiled_runtime_module_ = metal::compile_runtime_module();

  metal::KernelManager::Params params;
  params.compiled_runtime_module = compiled_runtime_module_.value();
  params.config = config;
  params.host_result_buffer = *result_buffer_ptr;
  params.mem_pool = memory_pool;
  params.profiler = profiler;
  metal_kernel_mgr_ = std::make_unique<metal::KernelManager>(std::move(params));
}

void MetalProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  (void)compile_snode_tree_types_impl(tree);
}

void MetalProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                              uint64 *result_buffer) {
  const auto &csnode_tree = compile_snode_tree_types_impl(tree);
  metal_kernel_mgr_->add_compiled_snode_tree(csnode_tree);
}

std::unique_ptr<AotModuleBuilder> MetalProgramImpl::make_aot_module_builder() {
  TI_ERROR_IF(compiled_snode_trees_.size() > 1,
              "AOT: only supports one SNodeTree");
  const auto fields =
      find_all_dense_snodes(compiled_snode_trees_[0].snode_descriptors);
  TI_ERROR_IF(!all_fields_are_dense(fields), "AOT: only supports dense field");
  return std::make_unique<metal::AotModuleBuilderImpl>(
      &(compiled_runtime_module_.value()), compiled_snode_trees_, fields,
      metal_kernel_mgr_->get_buffer_meta_data());
}

const metal::CompiledStructs &MetalProgramImpl::compile_snode_tree_types_impl(
    SNodeTree *tree) {
  TI_ASSERT_INFO(config->use_llvm,
                 "Metal arch requires that LLVM being enabled");
  auto *const root = tree->root();
  auto csnode_tree = metal::compile_structs(*root);
  compiled_snode_trees_.push_back(std::move(csnode_tree));
  return compiled_snode_trees_.back();
}

DeviceAllocation MetalProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  Device::AllocParams params;
  params.size = alloc_size;
  params.host_read = false;
  params.host_write = false;
  params.usage = AllocUsage::Storage;
  params.export_sharing = false;
  return metal_kernel_mgr_->allocate_memory(params);
}

}  // namespace lang
}  // namespace taichi
