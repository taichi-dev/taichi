// Codegen for the hierarchical data structure
#pragma once

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/backends/opengl/opengl_data_types.h"
#include "taichi/ir/snode.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

TLANG_NAMESPACE_BEGIN
namespace opengl {

class OpenglStructCompiler {
 public:
  using CompiledResult = opengl::StructCompiledResult;

  CompiledResult run(SNode &node);

 private:
  void collect_snodes(SNode &snode);
  void generate_types(const SNode &snode);
  size_t compute_snode_size(const SNode &sn);

  std::vector<SNode *> snodes_;
  std::unordered_map<SNodeId, SNodeInfo> snode_map_;
};

}  // namespace opengl
TLANG_NAMESPACE_END
