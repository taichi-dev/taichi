// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
int _msg_allocate_slot() {
  return atomicAdd(_msg_count_, 1);
}

void _msg_set(int msgid, int contid, int type, int value) {
  _msg_buf_[msgid].contents[contid] = value;
  _msg_buf_[msgid].type_bitmap_low |= (type & 1) << contid;
  _msg_buf_[msgid].type_bitmap_high |= (type >> 1) << contid;
}

void _msg_set_end(int msgid, int contid) {
  _msg_buf_[msgid].num_contents = contid;
}

void _msg_set_i32(int msgid, int contid, int value) {
  _msg_set(msgid, contid, 1, value);
}

void _msg_set_f32(int msgid, int contid, float value) {
  _msg_set(msgid, contid, 2, floatBitsToInt(value));
}

void _msg_set_str(int msgid, int contid, int stridx) {
  _msg_set(msgid, contid, 3, stridx);
}
)
