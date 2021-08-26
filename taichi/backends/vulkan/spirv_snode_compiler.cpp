#include "taichi/backends/vulkan/spirv_snode_compiler.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace spirv {

// Compile SNodes into Spirv-type-based struct
class SpirvSNodeCompiler {
 public:
  CompiledSpirvSNode run(IRBuilder *builder,
                         const CompiledSNodeStructs *compiled_structs) {
    CompiledSpirvSNode result;
    if (compiled_structs->root_size != 0) {
      result.root_stype = compute_snode_stype(
          builder, compiled_structs,
          compiled_structs->snode_descriptors.find(compiled_structs->root->id)
              ->second,
          &result.snode_id_struct_stype_tbl, &result.snode_id_array_stype_tbl);
    } else {  // Use an arbitary default type to skip empty root buffer
      result.root_stype = builder->i32_type();
    }
    return result;
  }

  SType compute_snode_stype(IRBuilder *ir_,
                            const CompiledSNodeStructs *compiled_structs,
                            const SNodeDescriptor &sn_desc,
                            SNodeSTypeTbl *snode_id_struct_stype_tbl_,
                            SNodeSTypeTbl *snode_id_array_stype_tbl_) {
    const auto &sn = sn_desc.snode;
    if (sn->is_place()) {
      return ir_->get_primitive_buffer_type(true, sn->dt);
    } else {
      SType sn_type = ir_->get_null_type();
      sn_type.snode_desc = sn_desc;
      sn_type.flag = TypeKind::kSNodeStruct;
      ir_->debug(spv::OpName, sn_type, sn->node_type_name);

      uint32_t cn_cnt = 0;
      for (const auto &ch : sn->ch) {
        const SNodeDescriptor &ch_desc =
            compiled_structs->snode_descriptors.find(ch->id)->second;
        const auto &ch_sn = ch_desc.snode;
        SType ch_type = compute_snode_stype(ir_, compiled_structs, ch_desc,
                                            snode_id_struct_stype_tbl_,
                                            snode_id_array_stype_tbl_);
        SType ch_type_array;

        if (!ch_sn->is_place()) {
          ch_type_array = ir_->get_null_type();
          ch_type_array.flag = TypeKind::kSNodeArray;
          ch_type_array.element_type_id = ch_type.id;

          Value length = ir_->int_immediate_number(
              ir_->i32_type(), ch_desc.cells_per_container_pot());
          ir_->declare_global(spv::OpTypeArray, ch_type_array, ch_type,
                              length);  // Length
          ir_->decorate(spv::OpDecorate, ch_type_array,
                        spv::DecorationArrayStride,
                        ch_desc.cell_stride);  // Stride
        } else {
          ch_type_array = ch_type;
        }
        ir_->decorate(spv::OpMemberDecorate, sn_type, cn_cnt++,
                      spv::DecorationOffset,
                      ch_desc.mem_offset_in_parent_cell);  // Offset
        sn_type.snode_child_type_id.push_back(ch_type_array.id);

        TI_ASSERT(snode_id_struct_stype_tbl_->find(ch_sn->id) ==
                  snode_id_struct_stype_tbl_->end());
        snode_id_struct_stype_tbl_->insert(
            std::make_pair(ch_sn->id, std::move(ch_type)));
        TI_ASSERT(snode_id_array_stype_tbl_->find(ch_sn->id) ==
                  snode_id_array_stype_tbl_->end());
        snode_id_array_stype_tbl_->insert(
            std::make_pair(ch_sn->id, std::move(ch_type_array)));
      }

      ir_->declare_global(spv::OpTypeStruct, sn_type,
                          sn_type.snode_child_type_id);
      return sn_type;
    }
  }
};

CompiledSpirvSNode compile_spirv_snode_structs(
    IRBuilder *builder,
    const CompiledSNodeStructs *compiled_structs) {
  SpirvSNodeCompiler compiler;
  return compiler.run(builder, compiled_structs);
}

}  // namespace spirv
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
