// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
double atomicAdd_extr_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _extr_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_extr_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicSub_extr_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _extr_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_extr_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMax_extr_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _extr_i64_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_extr_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMin_extr_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _extr_i64_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_extr_i64_[addr], old, new));
  return intBitsToFloat(old);
}
)
