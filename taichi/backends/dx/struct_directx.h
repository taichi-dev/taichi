#pragma once

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/ir/snode.h"
#include <vector>
#include <unordered_map>

TLANG_NAMESPACE_BEGIN
namespace dx {

// Copied from opengl_kernel_launcher.h
using SNodeId = std::string;
struct SNodeInfo {
  size_t stride;
  size_t length;
  std::vector<size_t> children_offsets;
  size_t elem_stride;
};
struct StructCompiledResult {
  std::unordered_map<SNodeId, SNodeInfo> snode_map;
  size_t root_size;
};

class DxStructCompiler {
public:
  StructCompiledResult Run(SNode* node);
  void CollectSNodes(SNode* snode);
  size_t ComputeSNodeSize(SNode *snode);
  void GenerateTypes(SNode *snode);

  int level;
  std::vector<SNode*> snodes_;
  std::unordered_map<SNodeId, SNodeInfo> snode_map_;
};

}
TLANG_NAMESPACE_END