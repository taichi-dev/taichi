// vim: ft=c
// clang-format off
#include "taichi/util/macros.h"
STR(
extern int RTi_max_i(int x, int y) {
  return x > y ? x : y;
}
extern int RTi_min_i(int x, int y) {
  return x < y ? x : y;
}
extern int RTi_max_f(float x, float y) {
  return x > y ? x : y;
}
extern int RTi_min_f(float x, float y) {
  return x < y ? x : y;
}
)
