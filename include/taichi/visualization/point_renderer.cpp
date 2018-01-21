/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/visualization/point_renderer.h>
#include <taichi/common/util.h>

#if TC_USE_OPENGL

TC_NAMESPACE_BEGIN

PointRenderer::PointRenderer(int max_size) : max_size(max_size) {
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * max_size, nullptr,
               GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
  glBindVertexArray(0);

  program = load_program("point", "point");
  CGL;
  setViewport(vec2(0), vec2(1));
}

void PointRenderer::render(vector<vec2> points, float point_size) {
  assert(points.size() <= max_size);
  glPointSize(point_size);

  glBindVertexArray(vao);
  glDisable(GL_DEPTH_TEST);
  for (auto &p : points) {
    p = (p - lower_left) / (upper_right - lower_left) * 2.0f - 1.0_f;
  }

  glUseProgram(program);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(points[0]) * points.size(),
                  &points[0]);

  glDrawArrays(GL_POINTS, 0, (GLsizei)points.size());
  glBindVertexArray(0);
  glUseProgram(0);
  CGL;
}

void PointRenderer::setViewport(vec2 lower_left, vec2 upper_right) {
  this->lower_left = lower_left;
  this->upper_right = upper_right;
}

TC_NAMESPACE_END

#endif
