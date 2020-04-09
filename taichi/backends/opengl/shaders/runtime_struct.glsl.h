// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
struct _RT_ {
  int rand_state;
  int unused1;
  bool bitmask[233]; /* TODO: use a actual value */
};
) "\n"
