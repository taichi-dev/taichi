// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
float atomicAdd_data_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _data_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_data_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicSub_data_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _data_i32_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_data_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMax_data_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _data_i32_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_data_i32_[addr], old, new));
  return intBitsToFloat(old);
}
float atomicMin_data_f32(int addr, float rhs) {
  int old, new, ret;
  do {
    old = _data_i32_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_data_i32_[addr], old, new));
  return intBitsToFloat(old);
}
)
