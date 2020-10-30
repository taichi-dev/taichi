// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
"#version 430 core\nprecision highp float;\n"
#define __GLSL__
#include "taichi/backends/opengl/shaders/runtime.h"
#undef __GLSL__
STR(
void _compute_indirect(
  int const_begin, int const_end,
  int gtmp_begin, int range_end,
  int SPT, int TPG) {
  // indirect work group size evaluator kernel
  _indirect_x_ = 128;
  _indirect_y_ = 1;
  _indirect_z_ = 1;
}
// get_indirect_evaluator() will prepend a main here, with template arguments
)
