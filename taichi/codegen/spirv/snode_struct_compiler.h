// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {
namespace spirv {

struct SNodeDescriptor {
  const SNode *snode = nullptr;
  // Stride (bytes) of a single cell.
  size_t cell_stride = 0;

  // Number of cells per container, padded to Power of Two (pot).
  size_t cells_per_container_pot() const;

  // Bytes of a single container.
  size_t container_stride = 0;

  // Total number of CELLS of this SNode, NOT padded to PoT.
  // For example, for a layout of
  // ti.root
  //   .dense(ti.ij, (3, 2))  // S1
  //   .dense(ti.ij, (5, 3))  // S2
  // |total_num_cells_from_root| for S2 is 3x2x5x3 = 90. That is, S2 has a total
  // of 90 cells. Note that the number of S2 (container) itself is 3x2=6!
  size_t total_num_cells_from_root = 0;
  // An SNode can have multiple number of components, where each component
  // starts at a fixed offset in its parent cell's memory.
  size_t mem_offset_in_parent_cell = 0;

  int axis_bits_sum[taichi_max_num_indices] = {0};
  int axis_start_bit[taichi_max_num_indices] = {0};

  SNode *get_child(int ch_i) const {
    return snode->ch[ch_i].get();
  }
};

using SNodeDescriptorsMap = std::unordered_map<int, SNodeDescriptor>;

struct CompiledSNodeStructs {
  // Root buffer size in bytes.
  size_t root_size;
  // Root SNode
  const SNode *root;
  // Map from SNode ID to its descriptor.
  SNodeDescriptorsMap snode_descriptors;
};

CompiledSNodeStructs compile_snode_structs(SNode &root);

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
