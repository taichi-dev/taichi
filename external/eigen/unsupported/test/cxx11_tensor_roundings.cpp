// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>


static void test_float_rounding()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 100.f;

  Tensor<float, 2> result = ftensor.round();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(result(i,j), numext::round(ftensor(i,j)));
    }
  }
}

static void test_float_flooring()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 100.f;

  Tensor<float, 2> result = ftensor.floor();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(result(i,j), numext::floor(ftensor(i,j)));
    }
  }
}

static void test_float_ceiling()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 100.f;

  Tensor<float, 2> result = ftensor.ceil();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(result(i,j), numext::ceil(ftensor(i,j)));
    }
  }
}

void test_cxx11_tensor_roundings()
{
   CALL_SUBTEST(test_float_rounding());
   CALL_SUBTEST(test_float_ceiling());
   CALL_SUBTEST(test_float_flooring());
}
