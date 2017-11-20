/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <taichi/common/config.h>
#include <taichi/math.h>

TC_NAMESPACE_BEGIN

template <int dim>
struct Element {
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  using Matrix = MatrixND<dim, real>;
  using MatrixP = MatrixND<dim + 1, real>;

  Vector v[dim];
  bool open_end[dim];

  Element() {
    for (int i = 0; i < dim; i++) {
      v[i] = Vector(0.0_f);
      open_end[i] = false;
    }
  }
};

template <int dim>
struct ElementMesh {
  using Elem = Element<dim>;
  std::vector<Elem> elements;
  using MatrixP = MatrixND<dim + 1, real>;
  MatrixP transform;

  void initialize(const Config &config) {

  }
};

TC_NAMESPACE_END
