// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Igor Babuschkin <igor@babuschk.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <numeric>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout, typename Type=float, bool Exclusive = false>
static void test_1d_scan()
{
  int size = 50;
  Tensor<Type, 1, DataLayout> tensor(size);
  tensor.setRandom();
  Tensor<Type, 1, DataLayout> result = tensor.cumsum(0, Exclusive);

  VERIFY_IS_EQUAL(tensor.dimension(0), result.dimension(0));

  float accum = 0;
  for (int i = 0; i < size; i++) {
    if (Exclusive) {
      VERIFY_IS_EQUAL(result(i), accum);
      accum += tensor(i);
    } else {
      accum += tensor(i);
      VERIFY_IS_EQUAL(result(i), accum);
    }
  }

  accum = 1;
  result = tensor.cumprod(0, Exclusive);
  for (int i = 0; i < size; i++) {
    if (Exclusive) {
      VERIFY_IS_EQUAL(result(i), accum);
      accum *= tensor(i);
    } else {
      accum *= tensor(i);
      VERIFY_IS_EQUAL(result(i), accum);
    }
  }
}

template <int DataLayout, typename Type=float>
static void test_4d_scan()
{
  int size = 5;
  Tensor<Type, 4, DataLayout> tensor(size, size, size, size);
  tensor.setRandom();

  Tensor<Type, 4, DataLayout> result(size, size, size, size);

  result = tensor.cumsum(0);
  float accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(i, 1, 2, 3);
    VERIFY_IS_EQUAL(result(i, 1, 2, 3), accum);
  }
  result = tensor.cumsum(1);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, i, 2, 3);
    VERIFY_IS_EQUAL(result(1, i, 2, 3), accum);
  }
  result = tensor.cumsum(2);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, 2, i, 3);
    VERIFY_IS_EQUAL(result(1, 2, i, 3), accum);
  }
  result = tensor.cumsum(3);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, 2, 3, i);
    VERIFY_IS_EQUAL(result(1, 2, 3, i), accum);
  }
}

template <int DataLayout>
static void test_tensor_maps() {
  int inputs[20];
  TensorMap<Tensor<int, 1, DataLayout> > tensor_map(inputs, 20);
  tensor_map.setRandom();

  Tensor<int, 1, DataLayout> result = tensor_map.cumsum(0);

  int accum = 0;
  for (int i = 0; i < 20; ++i) {
    accum += tensor_map(i);
    VERIFY_IS_EQUAL(result(i), accum);
  }
}

void test_cxx11_tensor_scan() {
  CALL_SUBTEST((test_1d_scan<ColMajor, float, true>()));
  CALL_SUBTEST((test_1d_scan<ColMajor, float, false>()));
  CALL_SUBTEST((test_1d_scan<RowMajor, float, true>()));
  CALL_SUBTEST((test_1d_scan<RowMajor, float, false>()));
  CALL_SUBTEST(test_4d_scan<ColMajor>());
  CALL_SUBTEST(test_4d_scan<RowMajor>());
  CALL_SUBTEST(test_tensor_maps<ColMajor>());
  CALL_SUBTEST(test_tensor_maps<RowMajor>());
}
