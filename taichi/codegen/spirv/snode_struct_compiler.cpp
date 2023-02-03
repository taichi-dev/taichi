#include "taichi/codegen/spirv/snode_struct_compiler.h"

namespace taichi::lang {
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
    /*
    result.type_factory = new tinyir::Block;
    result.root_type = construct(*result.type_factory, &root);
    */
    TI_TRACE("RootBuffer size={}", result.root_size);

    /*
    std::unique_ptr<tinyir::Block> b = ir_reduce_types(result.type_factory);

    TI_WARN("Original types:\n{}", ir_print_types(result.type_factory));

    TI_WARN("Reduced types:\n{}", ir_print_types(b.get()));
    */

    return result;
  }

 private:
  const tinyir::Type *construct(tinyir::Block &ir_module, SNode *sn) {
    const tinyir::Type *cell_type = nullptr;

    if (sn->is_place()) {
      // Each cell is a single Type
      cell_type = translate_ti_primitive(ir_module, sn->dt);
    } else {
      // Each cell is a struct
      std::vector<const tinyir::Type *> struct_elements;
      for (auto &ch : sn->ch) {
        const tinyir::Type *elem_type = construct(ir_module, ch.get());
        struct_elements.push_back(elem_type);
      }
      tinyir::Type *st = ir_module.emplace_back<StructType>(struct_elements);
      st->set_debug_name(
          fmt::format("{}_{}", snode_type_name(sn->type), sn->get_name()));
      cell_type = st;

      if (sn->type == SNodeType::pointer) {
        cell_type = ir_module.emplace_back<PhysicalPointerType>(cell_type);
      }
    }

    if (sn->num_cells_per_container == 1 || sn->is_scalar()) {
      return cell_type;
    } else {
      return ir_module.emplace_back<ArrayType>(cell_type,
                                               sn->num_cells_per_container);
    }
  }

  std::size_t compute_snode_size(SNode *sn) {
    const bool is_place = sn->is_place();

    SNodeDescriptor sn_desc;
    sn_desc.snode = sn;
    if (is_place) {
      sn_desc.cell_stride = data_type_size(sn->dt);
      sn_desc.container_stride = sn_desc.cell_stride;
    } else {
      // Sort by size, so that smaller subfields are placed first.
      // This accelerates Nvidia's GLSL compiler, as the compiler tries to
      // place all statically accessed fields
      std::vector<std::pair<size_t, int>> element_strides;
      int i = 0;
      for (auto &ch : sn->ch) {
        element_strides.push_back({compute_snode_size(ch.get()), i});
        i += 1;
      }
      std::sort(
          element_strides.begin(), element_strides.end(),
          [](const std::pair<size_t, int> &a, const std::pair<size_t, int> &b) {
            return a.first < b.first;
          });

      std::size_t cell_stride = 0;
      for (auto &[snode_size, i] : element_strides) {
        auto &ch = sn->ch[i];
        auto child_offset = cell_stride;
        auto *ch_snode = ch.get();
        cell_stride += snode_size;
        snode_descriptors_.find(ch_snode->id)
            ->second.mem_offset_in_parent_cell = child_offset;
        ch_snode->offset_bytes_in_parent_cell = child_offset;
      }
      sn_desc.cell_stride = cell_stride;

      if (sn->type == SNodeType::bitmasked) {
        size_t num_cells = sn_desc.snode->num_cells_per_container;
        size_t bitmask_num_words =
            num_cells % 32 == 0 ? (num_cells / 32) : (num_cells / 32 + 1);
        sn_desc.container_stride =
            cell_stride * num_cells + bitmask_num_words * 4;
      } else {
        sn_desc.container_stride =
            cell_stride * sn_desc.snode->num_cells_per_container;
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

    TI_TRACE("SNodeDescriptor");
    TI_TRACE("* snode={}", sn_desc.snode->id);
    TI_TRACE("* type={} (is_place={})", sn_desc.snode->node_type_name,
             is_place);
    TI_TRACE("* cell_stride={}", sn_desc.cell_stride);
    TI_TRACE("* num_cells_per_container={}",
             sn_desc.snode->num_cells_per_container);
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

CompiledSNodeStructs compile_snode_structs(SNode &root) {
  StructCompiler compiler;
  return compiler.run(root);
}

}  // namespace spirv
}  // namespace taichi::lang
