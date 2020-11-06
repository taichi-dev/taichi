// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
uvec4 _rand_;  // per-thread local variable

void _init_rand() {
  uint i = (54321u + gl_GlobalInvocationID.x) * (12345u + uint(_rand_state_[0]));
  _rand_.x = 123456789u * i * 1000000007u;
  _rand_.y = 362436069u;
  _rand_.z = 521288629u;
  _rand_.w = 88675123u;

  // Yes, this is not an atomic operation, but just fine since no matter
  // how `_rand_state_` changes, `gl_GlobalInvocationID.x` can still help
  // us to set different seeds for different threads.
  // Discussion: https://github.com/taichi-dev/taichi/pull/912#discussion_r419021918
  _rand_state_[0] += 1;
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
