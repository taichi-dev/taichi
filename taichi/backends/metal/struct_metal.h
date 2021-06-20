// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

struct SNodeDescriptor {
  const SNode *snode = nullptr;
  // Stride (bytes) of a single child element of this SNode.
  // TODO(k-ye): Rename to |per_ch_stride|.
  int element_stride = 0;
  // This is NOT the number of elements per |snode|. Instead, it's that number
  // padded to the closest power of two.
  // TODO(k-ye): Rename to |num_ch_slots|.
  int num_slots = 0;
  // Total bytes this SNode needs
  int stride = 0;
  // Total number of CHILDREN of this SNode, NOT padded to PoT.
  // For example, for a layout of
  // ti.root
  //   .dense(ti.ij, (3, 2))  // S1
  //   .dense(ti.ij, (5, 3))  // S2
  // |total_num_elems_from_root| for S2 is 3x2x5x3 = 90. That is, S2 has a total
  // of 90 children. Note that the number of S2 itself is 3x2=6. S2's
  // ListManager also has a size of 6!
  int total_num_elems_from_root = 0;
  // An SNode can have any number of children, where each child starts at a
  // fixed offset in its parent's memory.
  int mem_offset_in_parent = 0;

  int total_num_self_from_root(
      const std::unordered_map<int, SNodeDescriptor> &sn_descs) const;
};

using SNodeDescriptorsMap = std::unordered_map<int, SNodeDescriptor>;

// See SNodeDescriptor::total_num_self_from_root
int total_num_self_from_root(const SNodeDescriptorsMap &m, int snode_id);

struct CompiledStructs {
  // Source code of the SNode data structures compiled to Metal
  std::string snode_structs_source_code;
  // Runtime related source code
  std::string runtime_utils_source_code;
  // Type name of the generated root SNode.
  std::string root_snode_type_name;
  // Root buffer size in bytes.
  size_t root_size;
  // Runtime struct size.
  // A Runtime struct is generated dynamically, depending on the number of
  // SNodes. It looks like something below:
  // struct Runtime {
  //     SNodeMeta snode_metas[max_snodes];
  //     SNodeExtractors snode_extractors[max_snodes];
  //     ListManagerData snode_lists[max_snodes];
  //     uint32_t rand_seeds[kNumRandSeeds];
  // };
  //
  // |runtime_size| will be sizeof(Runtime), which is useful for allocating the
  // buffer memory.
  //
  // If |need_snode_lists_data| is true, the buffer will consist of two parts.
  // The first part, with size being |runtime_size|, is used to hold the Runtime
  // struct as expected. The second part is used as a kernel-side memory pool.
  //
  // |------ Runtime -----|--------------- Metal memory pool ---------------|
  // |<-- runtime_size -->|<------- decided by config, usually ~GB -------->|
  //
  // TODO(k-ye): See if Metal ArgumentBuffer can directly store the pointers.
  size_t runtime_size;
  // max(ID of Root or Dense Snode) + 1
  int max_snodes;
  // Map from SNode ID to its descriptor.
  SNodeDescriptorsMap snode_descriptors;
};

// Compile all snodes to Metal source code
CompiledStructs compile_structs(SNode &root);

}  // namespace metal
TLANG_NAMESPACE_END
