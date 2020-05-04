// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

struct SNodeDescriptor {
  const SNode *snode = nullptr;
  // Stride (bytes) of a single child element of this SNode.
  int element_stride = 0;
  // This is NOT the number of elements per |snode|. Instead, it's that number
  // padded to the closest power of two.
  int num_slots = 0;
  // Total bytes this SNode needs
  int stride = 0;
  // Total number of this SNodes, NOT padded to PoT.
  // For example, for a layout of
  // ti.root
  //   .dense(ti.ij, (3, 2))  // S1
  //   .dense(ti.ij, (5, 3))  // S2
  // |total_num_elems_from_root| for S2 is 3x2x5x3 = 90
  int total_num_elems_from_root = 0;
  // An SNode can have any number of children, where each child starts at a
  // fixed offset in its parent's memory.
  int mem_offset_in_parent = 0;
};

struct CompiledStructs {
  // Source code of the SNode data structures compiled to Metal
  std::string snode_structs_source_code;
  // Runtime related source code
  std::string runtime_utils_source_code;
  // Root buffer size in bytes.
  size_t root_size;
  // Runtime struct size.
  // A Runtime struct is generated dynamically, depending on the number of
  // SNodes. It looks like something below:
  // struct Runtime {
  //     SNodeMeta snode_metas[max_snodes];
  //     SNodeExtractors snode_extractors[max_snodes];
  //     ListManager snode_lists[max_snodes];
  //     uint32_t rand_seeds[kNumRandSeeds];
  // }
  // However, |runtime_size| is usually greater than sizeof(Runtime). That is
  // because the memory is divided into two parts. The first part of
  // sizeof(Runtime), as expected, is used to hold the Runtime struct. The
  // second part is used to hold the data of |snode_lists|.
  //
  // |---- Runtime ----|--------------- snode_lists data ---------------|
  // |<------------------------ runtime_size -------------------------->|
  //
  // The actual data address for the i-th ListManager is:
  // runtime memory address + list[i].mem_begin
  // TODO(k-ye): See if Metal ArgumentBuffer can directly store the pointers.
  size_t runtime_size;
  // max(ID of Root or Dense Snode) + 1
  int max_snodes;
  // Map from SNode ID to its descriptor.
  std::unordered_map<int, SNodeDescriptor> snode_descriptors;
};

// Compile all snodes to Metal source code
CompiledStructs compile_structs(SNode &root);

}  // namespace metal
TLANG_NAMESPACE_END
