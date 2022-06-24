// vim: ft=glsl
// clang-format off
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"

#ifdef TI_INSIDE_OPENGL_CODEGEN
#define OPENGL_BEGIN_PRINT_DEF constexpr auto kOpenGLPrintSourceCode =
#define OPENGL_END_PRINT_DEF ;
#else
static_assert(false, "Do not include");
#define OPENGL_BEGIN_PRINT_DEF
#define OPENGL_END_PRINT_DEF
#endif

OPENGL_BEGIN_PRINT_DEF
STR(
void _msg_set(int mid, int cid, int type, int val) {
  _msg_buf_[mid].contents[cid] = val;
  _msg_buf_[mid].type_bm_lo |= (type & 1) << cid;
  _msg_buf_[mid].type_bm_hi |= (type >> 1) << cid;
}

void _msg_set_end(int mid, int cid) {
  _msg_buf_[mid].num_contents = cid;
}

void _msg_set_i32(int mid, int cid, int val) {
  _msg_set(mid, cid, 1, val);
}

void _msg_set_f32(int mid, int cid, float val) {
  _msg_set(mid, cid, 2, floatBitsToInt(val));
}

void _msg_set_str(int mid, int cid, int stridx) {
  _msg_set(mid, cid, 3, stridx);
}
)
OPENGL_END_PRINT_DEF

#undef OPENGL_BEGIN_PRINT_DEF
#undef OPENGL_END_PRINT_DEF
// NOLINTEND(*)
