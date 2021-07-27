#define MAX_LIST (1024 * 256)  // * 4 = 1 MB

#ifdef TI_INSIDE_OPENGL_CODEGEN
#define OPENG_BEGIN_LISTMAN_DEF constexpr auto kOpenGLListmanSourceCode =
// clang-format off

#include "taichi/util/macros.h"
OPENG_BEGIN_LISTMAN_DEF
STR(
layout(std430, binding = 7) buffer listman {
  int _list_len_;
  int _list_[];
};
)"\n";
#undef OPENG_BEGIN_LISTMAN_DEF

// clang-format on
#else

TLANG_NAMESPACE_BEGIN

struct GLSLListman {
  int list_len;
  int list[MAX_LIST];
};

TLANG_NAMESPACE_END

#endif
