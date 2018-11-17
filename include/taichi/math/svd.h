/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "math.h"

TC_NAMESPACE_BEGIN

template <int dim, typename T>
extern void eigen_svd(const MatrixND<dim, T> &m,
                      MatrixND<dim, T> &u,
                      MatrixND<dim, T> &sig,
                      MatrixND<dim, T> &v);

template <int dim, typename T>
extern void imp_svd(const MatrixND<dim, T> &m,
                    MatrixND<dim, T> &u,
                    MatrixND<dim, T> &sig,
                    MatrixND<dim, T> &v);

template <int dim, typename T>
extern void svd(const MatrixND<dim, T> &m,
                MatrixND<dim, T> &u,
                MatrixND<dim, T> &sig,
                MatrixND<dim, T> &v);

template <int dim, typename T>
extern void svd_rot(const MatrixND<dim, T> &m,
                    MatrixND<dim, T> &u,
                    MatrixND<dim, T> &sig,
                    MatrixND<dim, T> &v);

template <int dim, typename T>
extern void polar_decomp(const MatrixND<dim, T> &A,
                         MatrixND<dim, T> &r,
                         MatrixND<dim, T> &s);

template <>
TC_FORCE_INLINE void polar_decomp<2, float32>(const MatrixND<2, float32> &m,
                                              MatrixND<2, float32> &R,
                                              MatrixND<2, float32> &S) {
  auto x = m(0, 0) + m(1, 1);
  auto y = m(1, 0) - m(0, 1);
  auto scale = 1.0f / std::sqrt(x * x + y * y);
  auto c = x * scale;
  auto s = y * scale;
  R = Matrix2(Vector2(c, s), Vector2(-s, c));
  S = transposed(R) * m;
}

template <>
TC_FORCE_INLINE void polar_decomp<2, float64>(const MatrixND<2, float64> &m,
                                              MatrixND<2, float64> &R,
                                              MatrixND<2, float64> &S) {
  auto x = m(0, 0) + m(1, 1);
  auto y = m(1, 0) - m(0, 1);
  auto scale = 1.0 / std::sqrt(x * x + y * y);
  auto c = x * scale;
  auto s = y * scale;
  R = Matrix2d(Vector2d(c, s), Vector2d(-s, c));
  S = transposed(R) * m;
}

template <int dim, typename T>
extern void qr_decomp(const MatrixND<dim, T> &A,
                      MatrixND<dim, T> &q,
                      MatrixND<dim, T> &r);

void svd_eigen2(void const *A_, void *u_, void *sig_, void *v_);

void svd_eigen3(void const *A_, void *u_, void *sig_, void *v_);

TC_NAMESPACE_END
