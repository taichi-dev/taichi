// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void product_selfadjoint(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, 1, MatrixType::RowsAtCompileTime> RowVectorType;

  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, Dynamic, RowMajor> RhsMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3;
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows),
             v3(rows);
  RowVectorType r1 = RowVectorType::Random(rows),
                r2 = RowVectorType::Random(rows);
  RhsMatrixType m4 = RhsMatrixType::Random(rows,10);

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>(),
         s3 = internal::random<Scalar>();

  m1 = (m1.adjoint() + m1).eval();

  // rank2 update
  m2 = m1.template triangularView<Lower>();
  m2.template selfadjointView<Lower>().rankUpdate(v1,v2);
  VERIFY_IS_APPROX(m2, (m1 + v1 * v2.adjoint()+ v2 * v1.adjoint()).template triangularView<Lower>().toDenseMatrix());

  m2 = m1.template triangularView<Upper>();
  m2.template selfadjointView<Upper>().rankUpdate(-v1,s2*v2,s3);
  VERIFY_IS_APPROX(m2, (m1 + (s3*(-v1)*(s2*v2).adjoint()+numext::conj(s3)*(s2*v2)*(-v1).adjoint())).template triangularView<Upper>().toDenseMatrix());

  m2 = m1.template triangularView<Upper>();
  m2.template selfadjointView<Upper>().rankUpdate(-s2*r1.adjoint(),r2.adjoint()*s3,s1);
  VERIFY_IS_APPROX(m2, (m1 + s1*(-s2*r1.adjoint())*(r2.adjoint()*s3).adjoint() + numext::conj(s1)*(r2.adjoint()*s3) * (-s2*r1.adjoint()).adjoint()).template triangularView<Upper>().toDenseMatrix());

  if (rows>1)
  {
    m2 = m1.template triangularView<Lower>();
    m2.block(1,1,rows-1,cols-1).template selfadjointView<Lower>().rankUpdate(v1.tail(rows-1),v2.head(cols-1));
    m3 = m1;
    m3.block(1,1,rows-1,cols-1) += v1.tail(rows-1) * v2.head(cols-1).adjoint()+ v2.head(cols-1) * v1.tail(rows-1).adjoint();
    VERIFY_IS_APPROX(m2, m3.template triangularView<Lower>().toDenseMatrix());
  }
}

void test_product_selfadjoint()
{
  int s = 0;
  for(int i = 0; i < g_repeat ; i++) {
    CALL_SUBTEST_1( product_selfadjoint(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( product_selfadjoint(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( product_selfadjoint(Matrix3d()) );
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2);
    CALL_SUBTEST_4( product_selfadjoint(MatrixXcf(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2);
    CALL_SUBTEST_5( product_selfadjoint(MatrixXcd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_6( product_selfadjoint(MatrixXd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_7( product_selfadjoint(Matrix<float,Dynamic,Dynamic,RowMajor>(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
