/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <taichi/math/array_2d.h>

TC_NAMESPACE_BEGIN

class ToneMapper : Unit {
 public:
  void initialize(const Config &config) override {}

  virtual Array2D<Vector3> apply(const Array2D<Vector3> &inp) {
    return Array2D<Vector3>(Vector2i(0, 0));
  }
};

TC_INTERFACE(ToneMapper);

TC_NAMESPACE_END