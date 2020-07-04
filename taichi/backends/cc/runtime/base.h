// vim: ft=c
// clang-format off
#include "taichi/util/macros.h"
STR(
) "\n#define " STR(
MTi_max(_, x, y) x > y ? x : y;
) "\n#define " STR(
MTi_min(_, x, y) x < y ? x : y;
) "\n#define " STR(
MTi_atomic_max(_, x, y) if (*x > y) *x = y;
) "\n#define " STR(
MTi_atomic_min(_, x, y) if (*x > y) *x = y;
) "\n" STR(
)
