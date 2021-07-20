#include "taichi/backends/vulkan/snode_struct_compiler.h"

#include "taichi/backends/vulkan/data_type_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {
namespace {

class StructCompiler {
 public:
  CompiledSNodeStructs run(const SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);

    CompiledSNodeStructs result;
    result.root_size = compute_snode_size(&root);
    result.snode_descriptors = std::move(snode_descriptors_);
    TI_INFO("Vulkan RootBuffer size={}", result.root_size);
    return result;
  }

 private:
  std::size_t compute_snode_size(const SNode *sn) {
    const bool is_place = sn->is_place();

    SNodeDescriptor sn_desc;
    sn_desc.snode = sn;
    if (is_place) {
      sn_desc.cell_stride = vk_data_type_size(sn->dt);
      sn_desc.container_stride = sn_desc.cell_stride;
    } else {
      std::size_t cell_stride = 0;
      for (const auto &ch : sn->ch) {
        const auto child_offset = cell_stride;
        const auto *ch_snode = ch.get();
        cell_stride += compute_snode_size(ch_snode);
        snode_descriptors_.find(ch_snode->id)
            ->second.mem_offset_in_parent_cell = child_offset;
      }
      sn_desc.cell_stride = cell_stride;
      sn_desc.container_stride =
          cell_stride * sn_desc.cells_per_container_pot();
    }

    sn_desc.total_num_cells_from_root = 1;
    for (const auto &e : sn->extractors) {
      // Note that the extractors are set in two places:
      // 1. When a new SNode is first defined
      // 2. StructCompiler::infer_snode_properties()
      // The second step is the finalized result.
      sn_desc.total_num_cells_from_root *= e.num_elements;
    }

    TI_INFO("SNodeDescriptor");
    TI_INFO("* snode={}", sn_desc.snode->id);
    TI_INFO("* type={} (is_place={})", sn_desc.snode->node_type_name, is_place);
    TI_INFO("* cell_stride={}", sn_desc.cell_stride);
    TI_INFO("* cells_per_container_pot={}", sn_desc.cells_per_container_pot());
    TI_INFO("* container_stride={}", sn_desc.container_stride);
    TI_INFO("* total_num_cells_from_root={}",
            sn_desc.total_num_cells_from_root);
    TI_INFO("");

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.container_stride;
  }

  SNodeDescriptorsMap snode_descriptors_;
};

}  // namespace

int SNodeDescriptor::cells_per_container_pot() const {
  // For root, |snode->n| is 0.
  const auto ty = snode->type;
  if (ty == SNodeType::root || ty == SNodeType::place) {
    return 1;
  }
  return snode->n;
}

CompiledSNodeStructs compile_snode_structs(const SNode &root) {
  StructCompiler compiler;
  return compiler.run(root);
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
