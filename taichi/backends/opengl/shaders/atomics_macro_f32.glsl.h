// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
"#define DEFINE_ATOMIC_F32_FUNCTIONS(NAME) "
STR(
float atomicAdd_##NAME##_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicSub_##NAME##_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMax_##NAME##_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i32_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMin_##NAME##_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _##NAME##_i32_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new));
  return intBitsToFloat(old);
}
\n
)
