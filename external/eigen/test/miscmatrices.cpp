// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void miscMatrices(const MatrixType& m)
{
  /* this test covers the following files:
     DiagonalMatrix.h Ones.h
  */
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  Index rows = m.rows();
  Index cols = m.cols();

  Index r = internal::random<Index>(0, rows-1), r2 = internal::random<Index>(0, rows-1), c = internal::random<Index>(0, cols-1);
  VERIFY_IS_APPROX(MatrixType::Ones(rows,cols)(r,c), static_cast<Scalar>(1));
  MatrixType m1 = MatrixType::Ones(rows,cols);
  VERIFY_IS_APPROX(m1(r,c), static_cast<Scalar>(1));
  VectorType v1 = VectorType::Random(rows);
  v1[0];
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
  square(v1.asDiagonal());
  if(r==r2) VERIFY_IS_APPROX(square(r,r2), v1[r]);
  else VERIFY_IS_MUCH_SMALLER_THAN(square(r,r2), static_cast<Scalar>(1));
  square = MatrixType::Zero(rows, rows);
  square.diagonal() = VectorType::Ones(rows);
  VERIFY_IS_APPROX(square, MatrixType::Identity(rows, rows));
}

void test_miscmatrices()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( miscMatrices(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( miscMatrices(Matrix4d()) );
    CALL_SUBTEST_3( miscMatrices(MatrixXcf(3, 3)) );
    CALL_SUBTEST_4( miscMatrices(MatrixXi(8, 12)) );
    CALL_SUBTEST_5( miscMatrices(MatrixXcd(20, 20)) );
  }
}
