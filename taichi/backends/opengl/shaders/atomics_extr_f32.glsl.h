// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
float atomicAdd_extr_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _extr_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_extr_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicSub_extr_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _extr_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_extr_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMax_extr_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _extr_i32_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_extr_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMin_extr_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _extr_i32_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_extr_i32_[addr], old, new));
  return intBitsToFloat(old);
}
)
