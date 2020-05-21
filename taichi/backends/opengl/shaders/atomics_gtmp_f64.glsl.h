// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
double atomicAdd_gtmp_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _gtmp_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) + rhs));
  } while (old != atomicCompSwap(_gtmp_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicSub_gtmp_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _gtmp_i64_[addr];
    new = floatBitsToInt((intBitsToFloat(old) - rhs));
  } while (old != atomicCompSwap(_gtmp_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMax_gtmp_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _gtmp_i64_[addr];
    new = floatBitsToInt(max(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_gtmp_i64_[addr], old, new));
  return intBitsToFloat(old);
}
double atomicMin_gtmp_f64(int addr, double rhs) {
  int old, new, ret;
  do {
    old = _gtmp_i64_[addr];
    new = floatBitsToInt(min(intBitsToFloat(old), rhs));
  } while (old != atomicCompSwap(_gtmp_i64_[addr], old, new));
  return intBitsToFloat(old);
}
)
