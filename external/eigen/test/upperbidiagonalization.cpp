// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/SVD>

template<typename MatrixType> void upperbidiag(const MatrixType& m)
{
  const typename MatrixType::Index rows = m.rows();
  const typename MatrixType::Index cols = m.cols();

  typedef Matrix<typename MatrixType::RealScalar, MatrixType::RowsAtCompileTime,  MatrixType::ColsAtCompileTime> RealMatrixType;
  typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime,  MatrixType::RowsAtCompileTime> TransposeMatrixType;

  MatrixType a = MatrixType::Random(rows,cols);
  internal::UpperBidiagonalization<MatrixType> ubd(a);
  RealMatrixType b(rows, cols);
  b.setZero();
  b.block(0,0,cols,cols) = ubd.bidiagonal();
  MatrixType c = ubd.householderU() * b * ubd.householderV().adjoint();
  VERIFY_IS_APPROX(a,c);
  TransposeMatrixType d = ubd.householderV() * b.adjoint() * ubd.householderU().adjoint();
  VERIFY_IS_APPROX(a.adjoint(),d);
}

void test_upperbidiagonalization()
{
  for(int i = 0; i < g_repeat; i++) {
   CALL_SUBTEST_1( upperbidiag(MatrixXf(3,3)) );
   CALL_SUBTEST_2( upperbidiag(MatrixXd(17,12)) );
   CALL_SUBTEST_3( upperbidiag(MatrixXcf(20,20)) );
   CALL_SUBTEST_4( upperbidiag(Matrix<std::complex<double>,Dynamic,Dynamic,RowMajor>(16,15)) );
   CALL_SUBTEST_5( upperbidiag(Matrix<float,6,4>()) );
   CALL_SUBTEST_6( upperbidiag(Matrix<float,5,5>()) );
   CALL_SUBTEST_7( upperbidiag(Matrix<double,4,3>()) );
  }
}
