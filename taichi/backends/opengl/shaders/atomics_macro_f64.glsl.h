// vim: ft=glsl
// clang-format off
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

#ifdef TI_INSIDE_OPENGL_CODEGEN
#define OPENGL_BEGIN_ATOMIC_F64_DEF constexpr auto kOpenGLAtomicF64SourceCode =
#define OPENGL_END_ATOMIC_F64_DEF ;
#else
static_assert(false, "Do not include");
#define OPENGL_BEGIN_ATOMIC_F64_DEF
#define OPENGL_END_ATOMIC_F64_DEF
#endif

OPENGL_BEGIN_ATOMIC_F64_DEF
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
OPENGL_END_ATOMIC_F64_DEF

#undef OPENGL_BEGIN_ATOMIC_F64_DEF
#undef OPENGL_END_ATOMIC_F64_DEF
// NOLINTEND(*)
