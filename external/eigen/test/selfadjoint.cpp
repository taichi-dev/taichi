// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_CHECK_STATIC_ASSERTIONS
#include "main.h"

// This file tests the basic selfadjointView API,
// the related products and decompositions are tested in specific files.

template<typename MatrixType> void selfadjoint(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols);

  m1.diagonal() = m1.diagonal().real().template cast<Scalar>();

  // check selfadjoint to dense
  m3 = m1.template selfadjointView<Upper>();
  VERIFY_IS_APPROX(MatrixType(m3.template triangularView<Upper>()), MatrixType(m1.template triangularView<Upper>()));
  VERIFY_IS_APPROX(m3, m3.adjoint());

  m3 = m1.template selfadjointView<Lower>();
  VERIFY_IS_APPROX(MatrixType(m3.template triangularView<Lower>()), MatrixType(m1.template triangularView<Lower>()));
  VERIFY_IS_APPROX(m3, m3.adjoint());

  m3 = m1.template selfadjointView<Upper>();
  m4 = m2;
  m4 += m1.template selfadjointView<Upper>();
  VERIFY_IS_APPROX(m4, m2+m3);

  m3 = m1.template selfadjointView<Lower>();
  m4 = m2;
  m4 -= m1.template selfadjointView<Lower>();
  VERIFY_IS_APPROX(m4, m2-m3);

  VERIFY_RAISES_STATIC_ASSERT(m2.template selfadjointView<StrictlyUpper>());
  VERIFY_RAISES_STATIC_ASSERT(m2.template selfadjointView<UnitLower>());
}

void bug_159()
{
  Matrix3d m = Matrix3d::Random().selfadjointView<Lower>();
  EIGEN_UNUSED_VARIABLE(m)
}

void test_selfadjoint()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    int s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);

    CALL_SUBTEST_1( selfadjoint(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( selfadjoint(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( selfadjoint(Matrix3cf()) );
    CALL_SUBTEST_4( selfadjoint(MatrixXcd(s,s)) );
    CALL_SUBTEST_5( selfadjoint(Matrix<float,Dynamic,Dynamic,RowMajor>(s, s)) );
    
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
  
  CALL_SUBTEST_1( bug_159() );
}
