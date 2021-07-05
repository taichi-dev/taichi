// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"

#ifdef TI_INSIDE_OPENGL_CODEGEN
#define OPENGL_BEGIN_ATOMIC_F32_DEF constexpr auto kOpenGLAtomicF32SourceCode =
#define OPENGL_END_ATOMIC_F32_DEF ;
#else
static_assert(false, "Do not include");
#define OPENGL_BEGIN_ATOMIC_F32_DEF
#define OPENGL_END_ATOMIC_F32_DEF
#endif

OPENGL_BEGIN_ATOMIC_F32_DEF
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
OPENGL_END_ATOMIC_F32_DEF

#undef OPENGL_BEGIN_ATOMIC_F32_DEF
#undef OPENGL_END_ATOMIC_F32_DEF
