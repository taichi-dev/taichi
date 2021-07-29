#pragma once

#include "taichi/backends/vulkan/spirv_header.h"
#include "taichi/backends/vulkan/spirv_ir_builder.h"

namespace taichi {
namespace lang {
namespace vulkan {

namespace spirv {

using SNodeSTypeTbl = std::unordered_map<int, SType>;

struct CompiledSpirvSNode {
  SType root_stype;

  // map from snode id to snode struct SType
  SNodeSTypeTbl snode_id_struct_stype_tbl;
  // map from snode id to snode array SType
  SNodeSTypeTbl snode_id_array_stype_tbl;

  SType query_snode_struct_stype(const int &id) const {
    return snode_id_struct_stype_tbl.find(id)->second;
  }
  SType query_snode_array_stype(const int &id) const {
    return snode_id_array_stype_tbl.find(id)->second;
  }
};

CompiledSpirvSNode compile_spirv_snode_structs(
    IRBuilder *builer,
    const CompiledSNodeStructs *compiled_structs);

}  // namespace spirv
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
