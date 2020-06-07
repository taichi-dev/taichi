// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
int _msg_allocate_slot() {
  return atomicAdd(_msg_count_, 1);
}

void _msg_set(int msgid, int contid, int type, int value) {
  int base = msgid << 5; // MSG_SIZE = (1 << 5)
  _mesg_i32_[base + contid] = value;
  _mesg_i32_[base + 30] |= (type & 1) << contid;
  _mesg_i32_[base + 31] |= (type >> 1) << contid;
}

void _msg_set_end(int msgid, int contid) {
  int base = msgid << 5;
  _mesg_i32_[base + 29] = contid;
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
