#define MAX_LIST (1024 * 256)  // * 4 = 1 MB

#ifdef __GLSL__
// clang-format off

#include "taichi/util/macros.h"
STR(
layout(std430, binding = 7) buffer listman {
  int _list_len_;
  int _list_[];
};
)"\n"

// clang-format on
#else

TLANG_NAMESPACE_BEGIN

struct GLSLListman {
  int list_len;
  int list[MAX_LIST];
};

TLANG_NAMESPACE_END

#endif
