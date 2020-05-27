#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

class SNode;

namespace opengl {

struct UsedFeature {
  bool random{false};
  bool argument{false};
  bool extra_arg{false};
  bool external_ptr{false};
  bool simulated_atomic_float{false};
  bool int64{false};
  bool global_temp{false};
  bool fast_pow{false};
};

using SNodeId = std::string;

struct SNodeInfo {
  size_t stride;
  size_t length;
  std::vector<size_t> children_offsets;
  size_t elem_stride;
};

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to GLSL
  std::unordered_map<SNodeId, SNodeInfo> snode_map;
  // Root buffer size in bytes.
  size_t root_size;
};

struct IOV {
  void *base;
  size_t size;
};

}  // namespace opengl

TLANG_NAMESPACE_END
