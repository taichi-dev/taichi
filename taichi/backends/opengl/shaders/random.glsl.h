// vim: ft=glsl
// clang-format off
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

#ifdef TI_INSIDE_OPENGL_CODEGEN
#define OPENGL_BEGIN_RANDOM_DEF constexpr auto kOpenGLRandomSourceCode =
#define OPENGL_END_RANDOM_DEF ;
#else
static_assert(false, "Do not include");
#define OPENGL_BEGIN_RANDOM_DEF
#define OPENGL_END_RANDOM_DEF
#endif

OPENGL_BEGIN_RANDOM_DEF
STR(
uvec4 _rand_;  // per-thread local variable

void _init_rand() {
  // ad-hoc: hope no kernel will use more than 1024 gtmp variables...
  // here we just use gtmp buffer for yield different rand state each launch
  int RAND_STATE = 1024;

  uint i = (7654321u + gl_GlobalInvocationID.x)
         * (1234567u + 9723451u * uint(_gtmp_i32_[RAND_STATE]));
  _rand_.x = 123456789u * i * 1000000007u;
  _rand_.y = 362436069u;
  _rand_.z = 521288629u;
  _rand_.w = 88675123u;

  // Yes, this is not an atomic operation, but just fine since no matter
  // how RAND_STATE changes, `gl_GlobalInvocationID.x` can still help
  // us to set different seeds for different threads.
  // Discussion: https://github.com/taichi-dev/taichi/pull/912#discussion_r419021918
  _gtmp_i32_[RAND_STATE] += 1;
}

uint _rand_u32() {
  uint t = _rand_.x ^ (_rand_.x << 11);
  _rand_.xyz = _rand_.yzw;
  _rand_.w = (_rand_.w ^ (_rand_.w >> 19)) ^ (t ^ (t >> 8));
  return _rand_.w * 1000000007u;
}

float _rand_f32() {
  return float(_rand_u32()) * (1.0 / 4294967296.0);
}

int _rand_i32() { return int(_rand_u32()); }
)
OPENGL_END_RANDOM_DEF

#undef OPENGL_BEGIN_RANDOM_DEF
#undef OPENGL_END_RANDOM_DEF
// NOLINTEND(*)
