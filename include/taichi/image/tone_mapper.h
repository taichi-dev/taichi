/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <taichi/math/array_2d.h>

TC_NAMESPACE_BEGIN

class ToneMapper : Unit {
 public:
  void initialize(const Config &config) override {
  }

  virtual Array2D<Vector3> apply(const Array2D<Vector3> &inp) {
    return Array2D<Vector3>(Vector2i(0, 0));
  }
};

TC_INTERFACE(ToneMapper);

TC_NAMESPACE_END