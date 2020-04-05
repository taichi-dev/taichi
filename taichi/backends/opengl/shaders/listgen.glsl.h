// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
void main()
{
  int tid = gl_GlobalInvocationID.x;
  _bitmask_[tid] = true;
}
)
