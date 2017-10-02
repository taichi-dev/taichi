/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>

#include <taichi/common/util.h>
#include <taichi/math/vector.h>

TC_NAMESPACE_BEGIN

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
  Vector2 dir = (b - a) / ab;
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
  static const Vector2 q(123532, 532421123);
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

inline Vector3 set_up(const Vector3 &a, const Vector3 &y) {
  Vector3 x, z;
  if (std::abs(y.y) > 1.0_f - eps) {
    x = Vector3(1, 0, 0);
  } else {
    x = normalize(cross(y, Vector3(0, 1, 0)));
  }
  z = cross(x, y);
  return a.x * x + a.y * y + a.z * z;
}

inline Vector3 random_diffuse(const Vector3 &normal, real u, real v) {
  if (u > v) {
    std::swap(u, v);
  }
  if (v < eps) {
    v = eps;
  }
  u /= v;
  real xz = v, y = sqrt(1 - v * v);
  real phi = u * pi * 2;
  return set_up(Vector3(xz * cos(phi), y, xz * sin(phi)), normal);
}

inline Vector3 random_diffuse(const Vector3 &normal) {
  return random_diffuse(normal, rand(), rand());
}

inline bool inside_unit_cube(const Vector3 &p) {
  return 0 <= p[0] && p[0] < 1 && 0 <= p[1] && p[1] < 1 && 0 <= p[2] &&
         p[2] < 1;
}

inline Vector3 sample_sphere(float u, float v) {
  float x = u * 2 - 1;
  float phi = v * 2 * pi;
  float yz = sqrt(1 - x * x);
  return Vector3(x, yz * cos(phi), yz * sin(phi));
}

inline Vector3 reflect(const Vector3 &d, const Vector3 &n) {
  return d - dot(d, n) * 2.0f * n;
}

TC_NAMESPACE_END
