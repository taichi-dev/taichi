/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#pragma once

#include <taichi/util.h>
#include <taichi/math/vector.h>
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

template <int n, typename T>
TC_FORCE_INLINE VectorND<n, T> from_eigen(
    Eigen::Matrix<T, n, 1> eigen_vec) {
  VectorND<n, T> ret;
  for (int i = 0; i < n; i++) {
    ret[i] = eigen_vec(i);
  }
  return ret;
}

template <int n, typename T>
TC_FORCE_INLINE Eigen::Matrix<T, n, 1> to_eigen(
    const VectorND<n, T> &taichi_vec) {
  Eigen::Matrix<T, n, 1> ret;
  for (int i = 0; i < n; i++) {
    ret(i) = taichi_vec[i];
  }
  return ret;
}

template <int n, typename T>
TC_FORCE_INLINE std::enable_if_t<n != 1, MatrixND<n, T>> from_eigen(
    Eigen::Matrix<T, n, n> eigen_vec) {
  MatrixND<n, T> ret;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      ret[j][i] = eigen_vec(i, j);
    }
  }
  return ret;
}

template <int n, typename T>
TC_FORCE_INLINE Eigen::Matrix<T, n, n> to_eigen(
    const MatrixND<n, T> &taichi_vec) {
  Eigen::Matrix<T, n, n> ret;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      ret(i, j) = taichi_vec[j][i];
    }
  }
  return ret;
}

TC_NAMESPACE_END
