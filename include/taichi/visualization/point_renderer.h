/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#if TC_USE_OPENGL

#include <vector>
#include <taichi/system/opengl.h>

TC_NAMESPACE_BEGIN

class PointRenderer {
  GLuint program, vao, vbo;
  int max_size;
  Vector2 lower_left, upper_right;

 public:
  PointRenderer(int max_size = 1048576);
  void render(std::vector<vec2> points, float point_size = 1.0_f);
  void setViewport(vec2 lower_left, vec2 upper_right);
};

TC_NAMESPACE_END

#endif
