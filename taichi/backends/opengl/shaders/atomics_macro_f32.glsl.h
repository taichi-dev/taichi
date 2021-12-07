// vim: ft=glsl
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

#ifndef TI_INSIDE_OPENGL_CODEGEN
static_assert(false, "Do not include");
#endif

#define GENERATE_OPENGL_ATOMIC_F32(NAME)                                 \
  constexpr auto kOpenGlAtomicF32Source_##NAME = STR(                    \
      float atomicAdd_##NAME##_f32(int addr, float rhs) {                \
        int old, new, ret;                                               \
        do {                                                             \
          old = _##NAME##_i32_[addr];                                    \
          new = floatBitsToInt((intBitsToFloat(old) + rhs));             \
        } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new)); \
        return intBitsToFloat(old);                                      \
      } float atomicSub_##NAME##_f32(int addr, float rhs) {              \
        int old, new, ret;                                               \
        do {                                                             \
          old = _##NAME##_i32_[addr];                                    \
          new = floatBitsToInt((intBitsToFloat(old) - rhs));             \
        } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new)); \
        return intBitsToFloat(old);                                      \
      } float atomicMax_##NAME##_f32(int addr, float rhs) {              \
        int old, new, ret;                                               \
        do {                                                             \
          old = _##NAME##_i32_[addr];                                    \
          new = floatBitsToInt(max(intBitsToFloat(old), rhs));           \
        } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new)); \
        return intBitsToFloat(old);                                      \
      } float atomicMin_##NAME##_f32(int addr, float rhs) {              \
        int old, new, ret;                                               \
        do {                                                             \
          old = _##NAME##_i32_[addr];                                    \
          new = floatBitsToInt(min(intBitsToFloat(old), rhs));           \
        } while (old != atomicCompSwap(_##NAME##_i32_[addr], old, new)); \
        return intBitsToFloat(old);                                      \
      });
// NOLINTEND(*)
