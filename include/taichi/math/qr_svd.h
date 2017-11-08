/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

void eigen_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v);

void eigen_svd(Matrix3 m, Matrix3 &u, Matrix3 &s, Matrix3 &v);

void imp_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v);

void imp_svd(Matrix3 m, Matrix3 &u, Matrix3 &s, Matrix3 &v);

void svd(Matrix2 m, Matrix2 &u, Matrix2 &sig, Matrix2 &v);

void svd(Matrix3 m, Matrix3 &u, Matrix3 &sig, Matrix3 &v);

void polar_decomp(Matrix2 A, Matrix2 &r, Matrix2 &s);

void polar_decomp(Matrix3 A, Matrix3 &r, Matrix3 &s);

void qr_decomp(Matrix2 A, Matrix2 &q, Matrix2 &r);

void qr_decomp(Matrix3 A, Matrix3 &q, Matrix3 &r);

void svd_eigen2(void const *A_, void *u_, void *sig_, void *v_);

void svd_eigen3(void const *A_, void *u_, void *sig_, void *v_);

TC_NAMESPACE_END
