// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void verifySizeOf(const MatrixType&)
{
  typedef typename MatrixType::Scalar Scalar;
  if (MatrixType::RowsAtCompileTime!=Dynamic && MatrixType::ColsAtCompileTime!=Dynamic)
    VERIFY_IS_EQUAL(std::ptrdiff_t(sizeof(MatrixType)),std::ptrdiff_t(sizeof(Scalar))*std::ptrdiff_t(MatrixType::SizeAtCompileTime));
  else
    VERIFY_IS_EQUAL(sizeof(MatrixType),sizeof(Scalar*) + 2 * sizeof(typename MatrixType::Index));
}

void test_sizeof()
{
  CALL_SUBTEST(verifySizeOf(Matrix<float, 1, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 2, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 3, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 4, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 5, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 6, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 7, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 8, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 9, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 10, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 11, 1>()) );
  CALL_SUBTEST(verifySizeOf(Array<float, 12, 1>()) );
  CALL_SUBTEST(verifySizeOf(Vector2d()) );
  CALL_SUBTEST(verifySizeOf(Vector4f()) );
  CALL_SUBTEST(verifySizeOf(Matrix4d()) );
  CALL_SUBTEST(verifySizeOf(Matrix<double, 4, 2>()) );
  CALL_SUBTEST(verifySizeOf(Matrix<bool, 7, 5>()) );
  CALL_SUBTEST(verifySizeOf(MatrixXcf(3, 3)) );
  CALL_SUBTEST(verifySizeOf(MatrixXi(8, 12)) );
  CALL_SUBTEST(verifySizeOf(MatrixXcd(20, 20)) );
  CALL_SUBTEST(verifySizeOf(Matrix<float, 100, 100>()) );
  
  VERIFY(sizeof(std::complex<float>) == 2*sizeof(float));
  VERIFY(sizeof(std::complex<double>) == 2*sizeof(double));
}
