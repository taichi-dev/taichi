#include "taichi/codegen/spirv/snode_struct_compiler.h"

namespace taichi {
namespace lang {
namespace spirv {
namespace {

class StructCompiler {
 public:
  CompiledSNodeStructs run(SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);

    CompiledSNodeStructs result;
    result.root = &root;
    result.root_size = compute_snode_size(&root);
    result.snode_descriptors = std::move(snode_descriptors_);
    TI_TRACE("RootBuffer size={}", result.root_size);
    return result;
  }

 private:
  std::size_t compute_snode_size(SNode *sn) {
    const bool is_place = sn->is_place();

    SNodeDescriptor sn_desc;
    sn_desc.snode = sn;
    if (is_place) {
      sn_desc.cell_stride = data_type_size(sn->dt);
      sn_desc.container_stride = sn_desc.cell_stride;
    } else {
      std::size_t cell_stride = 0;
      for (auto &ch : sn->ch) {
        auto child_offset = cell_stride;
        auto *ch_snode = ch.get();
        cell_stride += compute_snode_size(ch_snode);
        snode_descriptors_.find(ch_snode->id)
            ->second.mem_offset_in_parent_cell = child_offset;
      }
      sn_desc.cell_stride = cell_stride;

      if (sn->type == SNodeType::bitmasked) {
        size_t num_cells = sn_desc.cells_per_container_pot();
        size_t bitmask_num_words =
            num_cells % 32 == 0 ? (num_cells / 32) : (num_cells / 32 + 1);
        sn_desc.container_stride =
            cell_stride * num_cells + bitmask_num_words * 4;
      } else {
        sn_desc.container_stride =
            cell_stride * sn_desc.cells_per_container_pot();
      }
    }

    sn->cell_size_bytes = sn_desc.cell_stride;

    sn_desc.total_num_cells_from_root = 1;
    for (const auto &e : sn->extractors) {
      // Note that the extractors are set in two places:
      // 1. When a new SNode is first defined
      // 2. StructCompiler::infer_snode_properties()
      // The second step is the finalized result.
      sn_desc.total_num_cells_from_root *= e.num_elements_from_root;
    }

    // Sum the bits per axis
    SNode *snode_head = sn;
    do {
      for (int i = 0; i < taichi_max_num_indices; i++) {
        const AxisExtractor &extractor = snode_head->extractors[i];
        if (extractor.active) {
          sn_desc.axis_bits_sum[i] += extractor.num_bits;
        }
      }
    } while ((snode_head = snode_head->parent));
    // Find the start bit
    sn_desc.axis_start_bit[0] = 0;
    for (int i = 1; i < taichi_max_num_indices; i++) {
      sn_desc.axis_start_bit[i] =
          sn_desc.axis_bits_sum[i - 1] + sn_desc.axis_start_bit[i - 1];
    }
    TI_TRACE("Indices at SNode {}", sn->get_name());
    for (int i = 0; i < taichi_max_num_indices; i++) {
      TI_TRACE("Index {}: {}..{}", i, sn_desc.axis_start_bit[i],
               sn_desc.axis_start_bit[i] + sn_desc.axis_bits_sum[i]);
    }

    TI_TRACE("SNodeDescriptor");
    TI_TRACE("* snode={}", sn_desc.snode->id);
    TI_TRACE("* type={} (is_place={})", sn_desc.snode->node_type_name,
             is_place);
    TI_TRACE("* cell_stride={}", sn_desc.cell_stride);
    TI_TRACE("* cells_per_container_pot={}", sn_desc.cells_per_container_pot());
    TI_TRACE("* container_stride={}", sn_desc.container_stride);
    TI_TRACE("* total_num_cells_from_root={}",
             sn_desc.total_num_cells_from_root);
    TI_TRACE("");

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.container_stride;
  }

  SNodeDescriptorsMap snode_descriptors_;
};

}  // namespace

size_t SNodeDescriptor::cells_per_container_pot() const {
  return snode->num_cells_per_container;
}

CompiledSNodeStructs compile_snode_structs(SNode &root) {
  StructCompiler compiler;
  return compiler.run(root);
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
