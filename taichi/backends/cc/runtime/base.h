// vim: ft=c
// clang-format off
#include "taichi/util/macros.h"
"#include <stdio.h>\n"
"#include <stdlib.h>\n"
"#include <math.h>\n"
"\n" STR(
/* libc doesn't provide max/min for integers... sadly */
static inline int RTi_max(int x, int y) {
  return x > y ? x : y;
}
static inline int RTi_min(int x, int y) {
  return x < y ? x : y;
}
static inline long long RTi_llmax(long long x, long long y) {
  return x > y ? x : y;
}
static inline long long RTi_llmin(long long x, long long y) {
  return x < y ? x : y;
}

) "\n" STR(
)
