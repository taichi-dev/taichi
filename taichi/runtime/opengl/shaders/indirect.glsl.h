// vim: ft=glsl
// clang-format off
// NOLINTBEGIN(*)
#include "taichi/util/macros.h"
"#version 430 core\nprecision highp float;\n"
#define TI_INSIDE_OPENGL_CODEGEN
#define TI_OPENGL_NESTED_INCLUDE
#include "taichi/runtime/opengl/shaders/runtime.h"
#undef TI_OPENGL_NESTED_INCLUDE
#undef TI_INSIDE_OPENGL_CODEGEN
STR(
// taichi uses gtmp for storing dynamic range endpoints
layout(std430, binding = 1) buffer gtmp_i32 { int _gtmp_i32_[]; };

// indirect work group size evaluator kernel template
void _compute_indirect(
  int const_begin, int const_end,
  int range_begin, int range_end,
  int SPT, int TPG) {

  // dynamic range for
  if (const_begin == 0) {
    range_begin = _gtmp_i32_[range_begin >> 2];
  }
  if (const_end == 0) {
    range_end = _gtmp_i32_[range_end >> 2];
  }
  int nstrides = 1;
  if (range_end > range_begin) {
    nstrides = range_end - range_begin;
  }

  int nthreads = max((nstrides + SPT - 1) / SPT, 1);
  int nblocks = max((nthreads + TPG - 1) / TPG, 1);

  _indirect_x_ = nblocks;
  _indirect_y_ = 1;
  _indirect_z_ = 1;
}

// get_indirect_evaluator() will prepend a main here, with template arguments
)
// NOLINTEND(*)
