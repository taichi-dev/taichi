// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Ricard Marxer <email@ricardmarxer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <iostream>

using namespace std;

template<typename MatrixType> void reverse(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  Index rows = m.rows();
  Index cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::Random(rows, cols), m2;
  VectorType v1 = VectorType::Random(rows);

  MatrixType m1_r = m1.reverse();
  // Verify that MatrixBase::reverse() works
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_r(i, j), m1(rows - 1 - i, cols - 1 - j));
    }
  }

  Reverse<MatrixType> m1_rd(m1);
  // Verify that a Reverse default (in both directions) of an expression works
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_rd(i, j), m1(rows - 1 - i, cols - 1 - j));
    }
  }

  Reverse<MatrixType, BothDirections> m1_rb(m1);
  // Verify that a Reverse in both directions of an expression works
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_rb(i, j), m1(rows - 1 - i, cols - 1 - j));
    }
  }

  Reverse<MatrixType, Vertical> m1_rv(m1);
  // Verify that a Reverse in the vertical directions of an expression works
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_rv(i, j), m1(rows - 1 - i, j));
    }
  }

  Reverse<MatrixType, Horizontal> m1_rh(m1);
  // Verify that a Reverse in the horizontal directions of an expression works
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_rh(i, j), m1(i, cols - 1 - j));
    }
  }

  VectorType v1_r = v1.reverse();
  // Verify that a VectorType::reverse() of an expression works
  for ( int i = 0; i < rows; i++ ) {
    VERIFY_IS_APPROX(v1_r(i), v1(rows - 1 - i));
  }

  MatrixType m1_cr = m1.colwise().reverse();
  // Verify that PartialRedux::reverse() works (for colwise())
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_cr(i, j), m1(rows - 1 - i, j));
    }
  }

  MatrixType m1_rr = m1.rowwise().reverse();
  // Verify that PartialRedux::reverse() works (for rowwise())
  for ( int i = 0; i < rows; i++ ) {
    for ( int j = 0; j < cols; j++ ) {
      VERIFY_IS_APPROX(m1_rr(i, j), m1(i, cols - 1 - j));
    }
  }

  Scalar x = internal::random<Scalar>();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  m1.reverse()(r, c) = x;
  VERIFY_IS_APPROX(x, m1(rows - 1 - r, cols - 1 - c));
  
  m2 = m1;
  m2.reverseInPlace();
  VERIFY_IS_APPROX(m2,m1.reverse().eval());
  
  m2 = m1;
  m2.col(0).reverseInPlace();
  VERIFY_IS_APPROX(m2.col(0),m1.col(0).reverse().eval());
  
  m2 = m1;
  m2.row(0).reverseInPlace();
  VERIFY_IS_APPROX(m2.row(0),m1.row(0).reverse().eval());
  
  m2 = m1;
  m2.rowwise().reverseInPlace();
  VERIFY_IS_APPROX(m2,m1.rowwise().reverse().eval());
  
  m2 = m1;
  m2.colwise().reverseInPlace();
  VERIFY_IS_APPROX(m2,m1.colwise().reverse().eval());

  m1.colwise().reverse()(r, c) = x;
  VERIFY_IS_APPROX(x, m1(rows - 1 - r, c));

  m1.rowwise().reverse()(r, c) = x;
  VERIFY_IS_APPROX(x, m1(r, cols - 1 - c));
}

void test_array_reverse()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( reverse(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( reverse(Matrix2f()) );
    CALL_SUBTEST_3( reverse(Matrix4f()) );
    CALL_SUBTEST_4( reverse(Matrix4d()) );
    CALL_SUBTEST_5( reverse(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( reverse(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_7( reverse(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_8( reverse(Matrix<float, 100, 100>()) );
    CALL_SUBTEST_9( reverse(Matrix<float,Dynamic,Dynamic,RowMajor>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
#ifdef EIGEN_TEST_PART_3
  Vector4f x; x << 1, 2, 3, 4;
  Vector4f y; y << 4, 3, 2, 1;
  VERIFY(x.reverse()[1] == 3);
  VERIFY(x.reverse() == y);
#endif
}
