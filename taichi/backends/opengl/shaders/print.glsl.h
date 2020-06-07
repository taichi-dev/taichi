// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
void _msg_push(int type, int value) {
  int i = atomicAdd(_msg_count_, 1) << 1;
  _mesg_i32_[i + 0] = type;
  _mesg_i32_[i + 1] = value;
}

void _msg_push_i32(int value) {
  _msg_push(0, value);
}

void _msg_push_f32(float value) {
  _msg_push(1, floatBitsToInt(value));
}

void _msg_push_strid(int id) {
  _msg_push(2, id);
}
)
