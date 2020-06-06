// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
void _msg_push_i32(int x) {
  int i = atomicAdd(_msg_count_, 1);
  _mesg_i32_[i] = x;
}

void _msg_push_f32(int x) {
  _msg_push_i32(floatBitsToInt(x));
}
)
