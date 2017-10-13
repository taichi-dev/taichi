/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>
#include <taichi/common/interface.h>
#include <taichi/math/array_2d.h>

TC_NAMESPACE_BEGIN

class Texture : public Unit {
 public:
  virtual void initialize(const Config &config) {
  }

  virtual Vector4 sample(const Vector2 &coord) const {
    return sample(Vector3(coord.x, coord.y, 0.5f));
  }

  virtual Vector4 sample(const Vector3 &coord) const {
    error("no impl");
    return Vector4(0.0_f);
  }

  Vector3 sample3(const Vector2 &coord) const {
    Vector4 tmp = sample(coord);
    return Vector3(tmp.x, tmp.y, tmp.z);
  }

  Vector3 sample3(const Vector3 &coord) const {
    Vector4 tmp = sample(coord);
    return Vector3(tmp.x, tmp.y, tmp.z);
  }

  Array2D<Vector4> rasterize(Vector2i res) const {
    Array2D<Vector4> image(res);
    Vector2 inv_res(1.0 / res[0], 1.0 / res[1]);
    for (auto &ind : image.get_region()) {
      image[ind] = sample(ind.get_pos() * inv_res);
    }
    return image;
  }

  Array2D<Vector4> rasterize(int width, int height) const {
    return rasterize(Vector2i(width, height));
  }

  Array2D<Vector3> rasterize3(Vector2i res) const {
    Array2D<Vector3> image(res);
    Vector2 inv_res(1.0 / res[0], 1.0 / res[1]);
    for (auto &ind : image.get_region()) {
      Vector4 color = sample(ind.get_pos() * inv_res);
      image[ind] = Vector3(color.x, color.y, color.z);
    }
    return image;
  }

  Array2D<Vector3> rasterize3(int width, int height) const {
    return rasterize3(Vector2i(width, height));
  }
};

TC_INTERFACE(Texture);

TC_NAMESPACE_END
