/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/geometry/primitives.h>
#include <taichi/visual/sampler.h>
#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

class Camera : public Unit {
 public:
  virtual void initialize(const Config &config) {
    this->res = config.get_vec2i("res");
    this->aspect_ratio = (real)res[0] / res[1];
  }

  virtual Ray sample(Vector2 offset, Vector2 size, StateSequence &rand) {
    error("no impl");
    return Ray(Vector3(0), Vector3(0));
  }

  Vector3 get_origin() { return multiply_matrix4(transform, origin, 1); }

  Vector3 get_dir() { return multiply_matrix4(transform, dir, 0); }

  virtual void get_pixel_coordinate(Vector3 dir, real &u, real &v) {
    error("no impl");
  }

  virtual real get_pixel_scaling() { return 1.0f; }

  int get_width() const { return res[0]; }

  int get_height() const { return res[1]; }

 protected:
  Vector3 origin, look_at, up, right, dir;
  Matrix4 transform;
  Vector2i res;
  real aspect_ratio;

  void set_dir_and_right() {
    this->dir = normalize(look_at - origin);
    this->up = normalize(up - dir * dot(dir, up));
    this->right = cross(dir, up);
  }

  Vector2 random_offset(Vector2 offset, Vector2 size, real u, real v) {
    return Vector2(offset.x + u * size.x - 0.5f, offset.y + v * size.y - 0.5f);
  }
};

TC_INTERFACE(Camera);

TC_NAMESPACE_END
