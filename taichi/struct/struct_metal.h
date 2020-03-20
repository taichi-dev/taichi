// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

struct SNodeDescriptor {
  const SNode* snode = nullptr;
  int element_stride = 0;
  // This is not the number of elements per |snode|. Instead, it's that number
  // padded to the closest power of two.
  int num_slots = 0;
  int stride = 0;
  int total_num_elems_from_root = 0;
  int mem_offset_in_parent = 0;
};

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to Metal
  std::string snode_structs_source_code;
  std::string runtime_utils_source_code;
  std::string runtime_kernels_source_code;
  // Root buffer size in bytes.
  size_t root_size;
  size_t runtime_size;
  // max(ID of Root or Dense Snode) + 1
  int max_snodes;
  std::unordered_map<int, SNodeDescriptor> snode_descriptors;
};

// Compile all snodes to Metal source code
StructCompiledResult compile_structs(SNode &root);

}  // namespace metal
TLANG_NAMESPACE_END
