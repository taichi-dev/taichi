// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"

template<typename MatrixType> void product_notemporary(const MatrixType& m)
{
  /* This test checks the number of temporaries created
   * during the evaluation of a complex expression */
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> ColMajorMatrixType;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  ColMajorMatrixType m1 = MatrixType::Random(rows, cols),
                     m2 = MatrixType::Random(rows, cols),
                     m3(rows, cols);
  RowVectorType rv1 = RowVectorType::Random(rows), rvres(rows);
  ColVectorType cv1 = ColVectorType::Random(cols), cvres(cols);
  RowMajorMatrixType rm3(rows, cols);

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>(),
         s3 = internal::random<Scalar>();

  Index c0 = internal::random<Index>(4,cols-8),
        c1 = internal::random<Index>(8,cols-c0),
        r0 = internal::random<Index>(4,cols-8),
        r1 = internal::random<Index>(8,rows-r0);

  VERIFY_EVALUATION_COUNT( m3 = (m1 * m2.adjoint()), 1);
  VERIFY_EVALUATION_COUNT( m3 = (m1 * m2.adjoint()).transpose(), 1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = m1 * m2.adjoint(), 0);

  VERIFY_EVALUATION_COUNT( m3 = s1 * (m1 * m2.transpose()), 1);
//   VERIFY_EVALUATION_COUNT( m3 = m3 + s1 * (m1 * m2.transpose()), 1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = s1 * (m1 * m2.transpose()), 0);

  VERIFY_EVALUATION_COUNT( m3 = m3 + (m1 * m2.adjoint()), 1);
  VERIFY_EVALUATION_COUNT( m3 = m3 - (m1 * m2.adjoint()), 1);

  VERIFY_EVALUATION_COUNT( m3 = m3 + (m1 * m2.adjoint()).transpose(), 1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = m3 + m1 * m2.transpose(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() += m3 + m1 * m2.transpose(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() -= m3 + m1 * m2.transpose(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() =  m3 - m1 * m2.transpose(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() += m3 - m1 * m2.transpose(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() -= m3 - m1 * m2.transpose(), 0);

  VERIFY_EVALUATION_COUNT( m3.noalias() = s1 * m1 * s2 * m2.adjoint(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() = s1 * m1 * s2 * (m1*s3+m2*s2).adjoint(), 1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = (s1 * m1).adjoint() * s2 * m2, 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() += s1 * (-m1*s3).adjoint() * (s2 * m2 * s3), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() -= s1 * (m1.transpose() * m2), 0);

  VERIFY_EVALUATION_COUNT(( m3.block(r0,r0,r1,r1).noalias() += -m1.block(r0,c0,r1,c1) * (s2*m2.block(r0,c0,r1,c1)).adjoint() ), 0);
  VERIFY_EVALUATION_COUNT(( m3.block(r0,r0,r1,r1).noalias() -= s1 * m1.block(r0,c0,r1,c1) * m2.block(c0,r0,c1,r1) ), 0);

  // NOTE this is because the Block expression is not handled yet by our expression analyser
  VERIFY_EVALUATION_COUNT(( m3.block(r0,r0,r1,r1).noalias() = s1 * m1.block(r0,c0,r1,c1) * (s1*m2).block(c0,r0,c1,r1) ), 1);

  VERIFY_EVALUATION_COUNT( m3.noalias() -= (s1 * m1).template triangularView<Lower>() * m2, 0);
  VERIFY_EVALUATION_COUNT( rm3.noalias() = (s1 * m1.adjoint()).template triangularView<Upper>() * (m2+m2), 1);
  VERIFY_EVALUATION_COUNT( rm3.noalias() = (s1 * m1.adjoint()).template triangularView<UnitUpper>() * m2.adjoint(), 0);

  VERIFY_EVALUATION_COUNT( m3.template triangularView<Upper>() = (m1 * m2.adjoint()), 0);
  VERIFY_EVALUATION_COUNT( m3.template triangularView<Upper>() -= (m1 * m2.adjoint()), 0);

  // NOTE this is because the blas_traits require innerstride==1 to avoid a temporary, but that doesn't seem to be actually needed for the triangular products
  VERIFY_EVALUATION_COUNT( rm3.col(c0).noalias() = (s1 * m1.adjoint()).template triangularView<UnitUpper>() * (s2*m2.row(c0)).adjoint(), 1);

  VERIFY_EVALUATION_COUNT( m1.template triangularView<Lower>().solveInPlace(m3), 0);
  VERIFY_EVALUATION_COUNT( m1.adjoint().template triangularView<Lower>().solveInPlace(m3.transpose()), 0);

  VERIFY_EVALUATION_COUNT( m3.noalias() -= (s1 * m1).adjoint().template selfadjointView<Lower>() * (-m2*s3).adjoint(), 0);
  VERIFY_EVALUATION_COUNT( m3.noalias() = s2 * m2.adjoint() * (s1 * m1.adjoint()).template selfadjointView<Upper>(), 0);
  VERIFY_EVALUATION_COUNT( rm3.noalias() = (s1 * m1.adjoint()).template selfadjointView<Lower>() * m2.adjoint(), 0);

  // NOTE this is because the blas_traits require innerstride==1 to avoid a temporary, but that doesn't seem to be actually needed for the triangular products
  VERIFY_EVALUATION_COUNT( m3.col(c0).noalias() = (s1 * m1).adjoint().template selfadjointView<Lower>() * (-m2.row(c0)*s3).adjoint(), 1);
  VERIFY_EVALUATION_COUNT( m3.col(c0).noalias() -= (s1 * m1).adjoint().template selfadjointView<Upper>() * (-m2.row(c0)*s3).adjoint(), 1);

  VERIFY_EVALUATION_COUNT( m3.block(r0,c0,r1,c1).noalias() += m1.block(r0,r0,r1,r1).template selfadjointView<Upper>() * (s1*m2.block(r0,c0,r1,c1)), 0);
  VERIFY_EVALUATION_COUNT( m3.block(r0,c0,r1,c1).noalias() = m1.block(r0,r0,r1,r1).template selfadjointView<Upper>() * m2.block(r0,c0,r1,c1), 0);

  VERIFY_EVALUATION_COUNT( m3.template selfadjointView<Lower>().rankUpdate(m2.adjoint()), 0);

  // Here we will get 1 temporary for each resize operation of the lhs operator; resize(r1,c1) would lead to zero temporaries
  m3.resize(1,1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = m1.block(r0,r0,r1,r1).template selfadjointView<Lower>() * m2.block(r0,c0,r1,c1), 1);
  m3.resize(1,1);
  VERIFY_EVALUATION_COUNT( m3.noalias() = m1.block(r0,r0,r1,r1).template triangularView<UnitUpper>()  * m2.block(r0,c0,r1,c1), 1);

  // Zero temporaries for lazy products ...
  VERIFY_EVALUATION_COUNT( Scalar tmp = 0; tmp += Scalar(RealScalar(1)) /  (m3.transpose().lazyProduct(m3)).diagonal().sum(), 0 );

  // ... and even no temporary for even deeply (>=2) nested products
  VERIFY_EVALUATION_COUNT( Scalar tmp = 0; tmp += Scalar(RealScalar(1)) /  (m3.transpose() * m3).diagonal().sum(), 0 );
  VERIFY_EVALUATION_COUNT( Scalar tmp = 0; tmp += Scalar(RealScalar(1)) /  (m3.transpose() * m3).diagonal().array().abs().sum(), 0 );

  // Zero temporaries for ... CoeffBasedProductMode
  VERIFY_EVALUATION_COUNT( m3.col(0).template head<5>() * m3.col(0).transpose() + m3.col(0).template head<5>() * m3.col(0).transpose(), 0 );

  // Check matrix * vectors
  VERIFY_EVALUATION_COUNT( cvres.noalias() = m1 * cv1, 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() -= m1 * cv1, 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() -= m1 * m2.col(0), 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() -= m1 * rv1.adjoint(), 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() -= m1 * m2.row(0).transpose(), 0 );

  VERIFY_EVALUATION_COUNT( cvres.noalias() = (m1+m1) * cv1, 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() = (rm3+rm3) * cv1, 0 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() = (m1+m1) * (m1*cv1), 1 );
  VERIFY_EVALUATION_COUNT( cvres.noalias() = (rm3+rm3) * (m1*cv1), 1 );

  // Check outer products
  m3 = cv1 * rv1;
  VERIFY_EVALUATION_COUNT( m3.noalias() = cv1 * rv1, 0 );
  VERIFY_EVALUATION_COUNT( m3.noalias() = (cv1+cv1) * (rv1+rv1), 1 );
  VERIFY_EVALUATION_COUNT( m3.noalias() = (m1*cv1) * (rv1), 1 );
  VERIFY_EVALUATION_COUNT( m3.noalias() += (m1*cv1) * (rv1), 1 );
  VERIFY_EVALUATION_COUNT( rm3.noalias() = (cv1) * (rv1 * m1), 1 );
  VERIFY_EVALUATION_COUNT( rm3.noalias() -= (cv1) * (rv1 * m1), 1 );
  VERIFY_EVALUATION_COUNT( rm3.noalias() = (m1*cv1) * (rv1 * m1), 2 );
  VERIFY_EVALUATION_COUNT( rm3.noalias() += (m1*cv1) * (rv1 * m1), 2 );

  // Check nested products
  VERIFY_EVALUATION_COUNT( cvres.noalias() = m1.adjoint() * m1 * cv1, 1 );
  VERIFY_EVALUATION_COUNT( rvres.noalias() = rv1 * (m1 * m2.adjoint()), 1 );
}

void test_product_notemporary()
{
  int s;
  for(int i = 0; i < g_repeat; i++) {
    s = internal::random<int>(16,EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_1( product_notemporary(MatrixXf(s, s)) );
    CALL_SUBTEST_2( product_notemporary(MatrixXd(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(16,EIGEN_TEST_MAX_SIZE/2);
    CALL_SUBTEST_3( product_notemporary(MatrixXcf(s,s)) );
    CALL_SUBTEST_4( product_notemporary(MatrixXcd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
