/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>

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
extern void polar_decomp(const MatrixND<dim, T> &A,
                  MatrixND<dim, T> &r,
                  MatrixND<dim, T> &s);

template <int dim, typename T>
extern void qr_decomp(const MatrixND<dim, T> &A,
               MatrixND<dim, T> &q,
               MatrixND<dim, T> &r);

void svd_eigen2(void const *A_, void *u_, void *sig_, void *v_);

void svd_eigen3(void const *A_, void *u_, void *sig_, void *v_);

TC_NAMESPACE_END
