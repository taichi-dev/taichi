/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

class Rigid2D : public Unit {
 public:
  void initialize(const Config &config) override {}

  void step(real delta_t) {}
};

TC_NAMESPACE_END
