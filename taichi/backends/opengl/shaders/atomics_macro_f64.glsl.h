// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
"#define DEFINE_ATOMIC_F64_FUNCTIONS(NAME) "
STR(
double atomicAdd_##NAME_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_##NAME##_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicSub_##NAME##_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_##NAME##_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMax_##NAME##_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i64_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_##NAME##_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMin_##NAME##_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i64_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_##NAME##_i64_[addr], old, new));
  return intBitsToFloat(old);
}
\n
)
