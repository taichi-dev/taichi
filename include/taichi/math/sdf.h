/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

class SDF : public Unit {
 public:
  virtual void initialize(const Config &config) {}

  virtual real eval(const Vector3 &p) const { return 1; }
};

namespace sdf {
inline real mod(real a, real b) {
  return a - b * std::floor(a / b);
}

inline real cmod(real a, real b) {
  return a - b * (std::floor(a / b + 0.5f));
}

inline real cmod(real a, real b, int l, int r) {
  return a - b * (clamp((int)std::floor(a / b + 0.5f), l, r));
}

inline real sphere(const Vector3 &p, real radius) {
  return length(p) - radius;
}

inline real box(Vector3 p, const Vector3 &e) {
  p = Vector3(abs(p.x), abs(p.y), abs(p.z));
  p -= e;
  return length(Vector3(max(p.x, 0.0_f), max(p.y, 0.0_f), max(p.z, 0.0_f))) +
         std::min(0.0_f, p.max());
}

inline real cylinder(Vector3 p, real r, real h) {
  return std::max(length(Vector2(p.x, p.z)) - r, std::abs(p.y) - h);
}

inline real combine(real a, real b) {
  return std::min(a, b);
}

inline real intersection(real a, real b) {
  return std::max(a, b);
}

inline real combine_smooth(real a, real b, real r) {
  real m = std::min(a, b);
  if (a < r && b < r) {
    return std::min(m, r - length(Vector2(r - a, r - b)));
  } else {
    return m;
  }
}

inline real cut(real a, real b) {
  return std::max(a, -b);
}
}

TC_INTERFACE(SDF);

TC_NAMESPACE_END
