#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

class SNode;

namespace opengl {

struct UsedFeature {
  // types:
  bool simulated_atomic_float{false};
  bool int64{false};
  bool float64{false};

  // sparse:
  bool listman{false};

  // buffers:
  bool buf_args{false};
  bool buf_earg{false};
  bool buf_extr{false};
  bool buf_gtmp{false};

  // utilties:
  bool fast_pow{false};
  bool random{false};
  bool print{false};

  // extensions:
#define PER_OPENGL_EXTENSION(x) bool extension_##x{false};
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
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

enum class GLBufId {
  Root = 0,
  Runtime = 6,
  Listman = 7,
  Gtmp = 1,
  Args = 2,
  Earg = 3,
  Extr = 4,
};

struct IOV {
  void *base;
  size_t size;
};

}  // namespace opengl

TLANG_NAMESPACE_END
