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
  int old_val, new_val, ret;
  do {
    old_val = _##NAME##_i64_[addr];
    new_val = floatBitsToInt((intBitsToFloat(old_val) + rhs));
  } while (old_val != atomicCompSwap(_##NAME##_i64_[addr], old_val, new_val));
  return intBitsToFloat(old_val);
}
double atomicSub_##NAME##_f64(int addr, double rhs) {
  int old_val, new_val, ret;
  do {
    old_val = _##NAME##_i64_[addr];
    new_val = floatBitsToInt((intBitsToFloat(old_val) - rhs));
  } while (old_val != atomicCompSwap(_##NAME##_i64_[addr], old_val, new_val));
  return intBitsToFloat(old_val);
}
double atomicMax_##NAME##_f64(int addr, double rhs) {
  int old_val, new_val, ret;
  do {
    old_val = _##NAME##_i64_[addr];
    new_val = floatBitsToInt(max(intBitsToFloat(old_val), rhs));
  } while (old_val != atomicCompSwap(_##NAME##_i64_[addr], old_val, new_val));
  return intBitsToFloat(old_val);
}
double atomicMin_##NAME##_f64(int addr, double rhs) {
  int old_val, new_val, ret;
  do {
    old_val = _##NAME##_i64_[addr];
    new_val = floatBitsToInt(min(intBitsToFloat(old_val), rhs));
  } while (old_val != atomicCompSwap(_##NAME##_i64_[addr], old_val, new_val));
  return intBitsToFloat(old_val);
}
\n
)
OPENGL_END_ATOMIC_F64_DEF

#undef OPENGL_BEGIN_ATOMIC_F64_DEF
#undef OPENGL_END_ATOMIC_F64_DEF
// NOLINTEND(*)
