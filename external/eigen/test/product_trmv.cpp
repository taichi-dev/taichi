// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void trmv(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m3(rows, cols);
  VectorType v1 = VectorType::Random(rows);

  Scalar s1 = internal::random<Scalar>();

  m1 = MatrixType::Random(rows, cols);

  // check with a column-major matrix
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::Lower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::Upper>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitLower>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::UnitLower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpper>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::UnitUpper>() * v1, largerEps));

  // check conjugated and scalar multiple expressions (col-major)
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(((s1*m3).conjugate() * v1).isApprox((s1*m1).conjugate().template triangularView<Eigen::Lower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY((m3.conjugate() * v1.conjugate()).isApprox(m1.conjugate().template triangularView<Eigen::Upper>() * v1.conjugate(), largerEps));

  // check with a row-major matrix
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::Lower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::Upper>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpper>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::UnitLower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitLower>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::UnitUpper>() * v1, largerEps));

  // check conjugated and scalar multiple expressions (row-major)
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY((m3.adjoint() * v1).isApprox(m1.adjoint().template triangularView<Eigen::Lower>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY((m3.adjoint() * (s1*v1.conjugate())).isApprox(m1.adjoint().template triangularView<Eigen::Upper>() * (s1*v1.conjugate()), largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpper>();

  // check transposed cases:
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY((v1.transpose() * m3).isApprox(v1.transpose() * m1.template triangularView<Eigen::Lower>(), largerEps));
  VERIFY((v1.adjoint() * m3).isApprox(v1.adjoint() * m1.template triangularView<Eigen::Lower>(), largerEps));
  VERIFY((v1.adjoint() * m3.adjoint()).isApprox(v1.adjoint() * m1.template triangularView<Eigen::Lower>().adjoint(), largerEps));

  // TODO check with sub-matrices
}

void test_product_trmv()
{
  int s = 0;
  for(int i = 0; i < g_repeat ; i++) {
    CALL_SUBTEST_1( trmv(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( trmv(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( trmv(Matrix3d()) );
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2);
    CALL_SUBTEST_4( trmv(MatrixXcf(s,s)) );
    CALL_SUBTEST_5( trmv(MatrixXcd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_6( trmv(Matrix<float,Dynamic,Dynamic,RowMajor>(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
