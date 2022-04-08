// vim: ft=glsl
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

#ifndef TI_INSIDE_OPENGL_CODEGEN
static_assert(false, "Do not include");
#endif

#define GENERATE_OPENGL_ATOMIC_F32(NAME)                                  \
  constexpr auto kOpenGlAtomicF32Source_##NAME = STR(                     \
      float atomicAdd_##NAME##_f32(int addr, float rhs) {                 \
        int old_val, new_val, ret;                                        \
        do {                                                              \
          old_val = _##NAME##_i32_[addr];                                 \
          new_val = floatBitsToInt((intBitsToFloat(old_val) + rhs));      \
        } while (old_val !=                                               \
                 atomicCompSwap(_##NAME##_i32_[addr], old_val, new_val)); \
        return intBitsToFloat(old_val);                                   \
      } float atomicSub_##NAME##_f32(int addr, float rhs) {               \
        int old_val, new_val, ret;                                        \
        do {                                                              \
          old_val = _##NAME##_i32_[addr];                                 \
          new_val = floatBitsToInt((intBitsToFloat(old_val) - rhs));      \
        } while (old_val !=                                               \
                 atomicCompSwap(_##NAME##_i32_[addr], old_val, new_val)); \
        return intBitsToFloat(old_val);                                   \
      } float atomicMax_##NAME##_f32(int addr, float rhs) {               \
        int old_val, new_val, ret;                                        \
        do {                                                              \
          old_val = _##NAME##_i32_[addr];                                 \
          new_val = floatBitsToInt(max(intBitsToFloat(old_val), rhs));    \
        } while (old_val !=                                               \
                 atomicCompSwap(_##NAME##_i32_[addr], old_val, new_val)); \
        return intBitsToFloat(old_val);                                   \
      } float atomicMin_##NAME##_f32(int addr, float rhs) {               \
        int old_val, new_val, ret;                                        \
        do {                                                              \
          old_val = _##NAME##_i32_[addr];                                 \
          new_val = floatBitsToInt(min(intBitsToFloat(old_val), rhs));    \
        } while (old_val !=                                               \
                 atomicCompSwap(_##NAME##_i32_[addr], old_val, new_val)); \
        return intBitsToFloat(old_val);                                   \
      });
// NOLINTEND(*)
