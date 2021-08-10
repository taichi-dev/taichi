// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>


static void test_empty_tensor()
{
  Tensor<float, 2> source;
  Tensor<float, 2> tgt1 = source;
  Tensor<float, 2> tgt2(source);
  Tensor<float, 2> tgt3;
  tgt3 = tgt1;
  tgt3 = tgt2;
}

static void test_empty_fixed_size_tensor()
{
  TensorFixedSize<float, Sizes<0> > source;
  TensorFixedSize<float, Sizes<0> > tgt1 = source;
  TensorFixedSize<float, Sizes<0> > tgt2(source);
  TensorFixedSize<float, Sizes<0> > tgt3;
  tgt3 = tgt1;
  tgt3 = tgt2;
}


void test_cxx11_tensor_empty()
{
   CALL_SUBTEST(test_empty_tensor());
   CALL_SUBTEST(test_empty_fixed_size_tensor());
}
