#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

class SNode;

namespace opengl {

constexpr int taichi_opengl_earg_base = taichi_max_num_args * sizeof(uint64_t);

struct UsedFeature {
  // types:
  bool simulated_atomic_float{false};
  bool int32{false};
  bool float32{false};
  bool int64{false};
  bool uint32{false};
  bool uint64{false};
  bool float64{false};

  // buffers:
  bool buf_args{false};
  bool buf_earg{false};
  bool buf_extr{false};
  bool buf_gtmp{false};
  bool buf_retr{false};

  // utilties:
  bool fast_pow{false};
  bool random{false};
  bool print{false};
  bool reduction{false};

  // extensions:
#define PER_OPENGL_EXTENSION(x) bool extension_##x{false};
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
};

enum class GLBufId {
  Root = 0,
  Gtmp = 1,
  Args = 2,
  Retr = 3,
  Extr = 4,
  Runtime = 5,
};

struct IOV {
  void *base{nullptr};
  size_t size{0};
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
