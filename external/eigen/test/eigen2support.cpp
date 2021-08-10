// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN2_SUPPORT

#include "main.h"

template<typename MatrixType> void eigen2support(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  Scalar  s1 = internal::random<Scalar>(),
          s2 = internal::random<Scalar>();

  // scalar addition
  VERIFY_IS_APPROX(m1.cwise() + s1, s1 + m1.cwise());
  VERIFY_IS_APPROX(m1.cwise() + s1, MatrixType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX((m1*Scalar(2)).cwise() - s2, (m1+m1) - MatrixType::Constant(rows,cols,s2) );
  m3 = m1;
  m3.cwise() += s2;
  VERIFY_IS_APPROX(m3, m1.cwise() + s2);
  m3 = m1;
  m3.cwise() -= s1;
  VERIFY_IS_APPROX(m3, m1.cwise() - s1);

  VERIFY_IS_EQUAL((m1.corner(TopLeft,1,1)), (m1.block(0,0,1,1)));
  VERIFY_IS_EQUAL((m1.template corner<1,1>(TopLeft)), (m1.template block<1,1>(0,0)));
  VERIFY_IS_EQUAL((m1.col(0).start(1)), (m1.col(0).segment(0,1)));
  VERIFY_IS_EQUAL((m1.col(0).template start<1>()), (m1.col(0).segment(0,1)));
  VERIFY_IS_EQUAL((m1.col(0).end(1)), (m1.col(0).segment(rows-1,1)));
  VERIFY_IS_EQUAL((m1.col(0).template end<1>()), (m1.col(0).segment(rows-1,1)));
  
  using std::cos;
  using numext::real;
  using numext::abs2;
  VERIFY_IS_EQUAL(ei_cos(s1), cos(s1));
  VERIFY_IS_EQUAL(ei_real(s1), real(s1));
  VERIFY_IS_EQUAL(ei_abs2(s1), abs2(s1));

  m1.minor(0,0);
}

void test_eigen2support()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( eigen2support(Matrix<double,1,1>()) );
    CALL_SUBTEST_2( eigen2support(MatrixXd(1,1)) );
    CALL_SUBTEST_4( eigen2support(Matrix3f()) );
    CALL_SUBTEST_5( eigen2support(Matrix4d()) );
    CALL_SUBTEST_2( eigen2support(MatrixXf(200,200)) );
    CALL_SUBTEST_6( eigen2support(MatrixXcd(100,100)) );
  }
}
