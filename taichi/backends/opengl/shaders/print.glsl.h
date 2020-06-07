// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
int _msg_allocate_slot() {
  return atomicAdd(_msg_count_, 1) << 5; // MAX_CONTENTS_PER_PRINT = (1 << 5)
}

void _msg_set(int m, int type, int value) {
  m = m << 1;
  _mesg_i32_[m + 0] = type;
  _mesg_i32_[m + 1] = value;
}

void _msg_set_i32(int m, int value) {
  _msg_set(m, -1, value);
}

void _msg_set_f32(int m, float value) {
  _msg_set(m, -2, floatBitsToInt(value));
}

void _msg_set_str(int m, int stridx) {
  _msg_set(m, -3, stridx);
}
)
