// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
uvec4 _rand_;

void _init_rand() {
  uint i = (54321 + gl_GlobalInvocationID.x) * (12345 + _states_[0]);
  _rand_.x = 123456789 * i * 1000000007;
  _rand_.y = 362436069;
  _rand_.z = 521288629;
  _rand_.w = 88675123;
  _states_[0] += 1;
}

uint _rand_u32() {
  uint t = _rand_.x ^ (_rand_.x << 11);
  _rand_.xyz = _rand_.yzw;
  _rand_.w = (_rand_.w ^ (_rand_.w >> 19)) ^ (t ^ (t >> 8));
  return _rand_.w * 1000000007;
}

float _rand_f32() {
  return float(_rand_u32()) * (1.0 / 4294967296.0);
}

double _rand_f64() { return double(_rand_f32()); }

int _rand_i32() { return int(_rand_u32()); }
)
