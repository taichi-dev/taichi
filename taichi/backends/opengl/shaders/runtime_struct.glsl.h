// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
struct _RT_ {
  int rand_state;
  int unused1;
  int bitmask[64]; /* TODO: use a actual size */
};
) "\n"
