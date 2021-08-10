// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/LU>

template<typename MatrixType> void determinant(const MatrixType& m)
{
  /* this test covers the following files:
     Determinant.h
  */
  Index size = m.rows();

  MatrixType m1(size, size), m2(size, size);
  m1.setRandom();
  m2.setRandom();
  typedef typename MatrixType::Scalar Scalar;
  Scalar x = internal::random<Scalar>();
  VERIFY_IS_APPROX(MatrixType::Identity(size, size).determinant(), Scalar(1));
  VERIFY_IS_APPROX((m1*m2).eval().determinant(), m1.determinant() * m2.determinant());
  if(size==1) return;
  Index i = internal::random<Index>(0, size-1);
  Index j;
  do {
    j = internal::random<Index>(0, size-1);
  } while(j==i);
  m2 = m1;
  m2.row(i).swap(m2.row(j));
  VERIFY_IS_APPROX(m2.determinant(), -m1.determinant());
  m2 = m1;
  m2.col(i).swap(m2.col(j));
  VERIFY_IS_APPROX(m2.determinant(), -m1.determinant());
  VERIFY_IS_APPROX(m2.determinant(), m2.transpose().determinant());
  VERIFY_IS_APPROX(numext::conj(m2.determinant()), m2.adjoint().determinant());
  m2 = m1;
  m2.row(i) += x*m2.row(j);
  VERIFY_IS_APPROX(m2.determinant(), m1.determinant());
  m2 = m1;
  m2.row(i) *= x;
  VERIFY_IS_APPROX(m2.determinant(), m1.determinant() * x);
  
  // check empty matrix
  VERIFY_IS_APPROX(m2.block(0,0,0,0).determinant(), Scalar(1));
}

void test_determinant()
{
  for(int i = 0; i < g_repeat; i++) {
    int s = 0;
    CALL_SUBTEST_1( determinant(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( determinant(Matrix<double, 2, 2>()) );
    CALL_SUBTEST_3( determinant(Matrix<double, 3, 3>()) );
    CALL_SUBTEST_4( determinant(Matrix<double, 4, 4>()) );
    CALL_SUBTEST_5( determinant(Matrix<std::complex<double>, 10, 10>()) );
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
    CALL_SUBTEST_6( determinant(MatrixXd(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
