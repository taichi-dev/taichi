/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>

#include "taichi/common/core.h"
#include "linalg.h"

TI_NAMESPACE_BEGIN

inline bool intersect(const Vector2 &a,
                      const Vector2 &b,
                      const Vector2 &c,
                      const Vector2 &d) {
  if (cross(c - a, b - a) * cross(b - a, d - a) > 0 &&
      cross(a - d, c - d) * cross(c - d, b - d) > 0) {
    return true;
  } else {
    return false;
  }
}

inline real nearest_distance(const Vector2 &p,
                             const Vector2 &a,
                             const Vector2 &b) {
  real ab = length(a - b);
  Vector2 dir = normalized(b - a);
  real pos = clamp(dot(p - a, dir), 0.0_f, ab);
  return length(a + pos * dir - p);
}

inline real nearest_distance(const Vector2 &p,
                             const std::vector<Vector2> &polygon) {
  real dist = std::numeric_limits<float>::infinity();
  for (int i = 0; i < (int)polygon.size(); i++) {
    dist = std::min(dist, nearest_distance(p, polygon[i],
                                           polygon[(i + 1) % polygon.size()]));
  }
  return dist;
}

inline bool inside_polygon(const Vector2 &p,
                           const std::vector<Vector2> &polygon) {
  int count = 0;
  static const Vector2 q(123532_f, 532421123_f);
  for (int i = 0; i < (int)polygon.size(); i++) {
    count += intersect(p, q, polygon[i], polygon[(i + 1) % polygon.size()]);
  }
  return count % 2 == 1;
}

inline std::vector<Vector2> points_inside_polygon(
    std::vector<float> x_range,
    std::vector<float> y_range,
    const std::vector<Vector2> &polygon) {
  std::vector<Vector2> ret;
  for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
    for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
      Vector2 p(x, y);
      if (inside_polygon(p, polygon)) {
        ret.push_back(p);
      }
    }
  }
  return ret;
}

inline std::vector<Vector2> points_inside_sphere(std::vector<float> x_range,
                                                 std::vector<float> y_range,
                                                 const Vector2 &center,
                                                 float radius) {
  std::vector<Vector2> ret;
  for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
    for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
      Vector2 p(x, y);
      if (length(p - center) < radius) {
        ret.push_back(p);
      }
    }
  }
  return ret;
}

TI_NAMESPACE_END
