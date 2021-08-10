// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/LU>

template<typename MatrixType> void inverse(const MatrixType& m)
{
  using std::abs;
  /* this test covers the following files:
     Inverse.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;

  MatrixType m1(rows, cols),
             m2(rows, cols),
             identity = MatrixType::Identity(rows, rows);
  createRandomPIMatrixOfRank(rows,rows,rows,m1);
  m2 = m1.inverse();
  VERIFY_IS_APPROX(m1, m2.inverse() );

  VERIFY_IS_APPROX((Scalar(2)*m2).inverse(), m2.inverse()*Scalar(0.5));

  VERIFY_IS_APPROX(identity, m1.inverse() * m1 );
  VERIFY_IS_APPROX(identity, m1 * m1.inverse() );

  VERIFY_IS_APPROX(m1, m1.inverse().inverse() );

  // since for the general case we implement separately row-major and col-major, test that
  VERIFY_IS_APPROX(MatrixType(m1.transpose().inverse()), MatrixType(m1.inverse().transpose()));

#if !defined(EIGEN_TEST_PART_5) && !defined(EIGEN_TEST_PART_6)
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;
  
  //computeInverseAndDetWithCheck tests
  //First: an invertible matrix
  bool invertible;
  Scalar det;

  m2.setZero();
  m1.computeInverseAndDetWithCheck(m2, det, invertible);
  VERIFY(invertible);
  VERIFY_IS_APPROX(identity, m1*m2);
  VERIFY_IS_APPROX(det, m1.determinant());

  m2.setZero();
  m1.computeInverseWithCheck(m2, invertible);
  VERIFY(invertible);
  VERIFY_IS_APPROX(identity, m1*m2);

  //Second: a rank one matrix (not invertible, except for 1x1 matrices)
  VectorType v3 = VectorType::Random(rows);
  MatrixType m3 = v3*v3.transpose(), m4(rows,cols);
  m3.computeInverseAndDetWithCheck(m4, det, invertible);
  VERIFY( rows==1 ? invertible : !invertible );
  VERIFY_IS_MUCH_SMALLER_THAN(abs(det-m3.determinant()), RealScalar(1));
  m3.computeInverseWithCheck(m4, invertible);
  VERIFY( rows==1 ? invertible : !invertible );
  
  // check with submatrices
  {
    Matrix<Scalar, MatrixType::RowsAtCompileTime+1, MatrixType::RowsAtCompileTime+1, MatrixType::Options> m5;
    m5.setRandom();
    m5.topLeftCorner(rows,rows) = m1;
    m2 = m5.template topLeftCorner<MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime>().inverse();
    VERIFY_IS_APPROX( (m5.template topLeftCorner<MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime>()), m2.inverse() );
  }
#endif

  // check in-place inversion
  if(MatrixType::RowsAtCompileTime>=2 && MatrixType::RowsAtCompileTime<=4)
  {
    // in-place is forbidden
    VERIFY_RAISES_ASSERT(m1 = m1.inverse());
  }
  else
  {
    m2 = m1.inverse();
    m1 = m1.inverse();
    VERIFY_IS_APPROX(m1,m2);
  }
}

template<typename Scalar>
void inverse_zerosized()
{
  Matrix<Scalar,Dynamic,Dynamic> A(0,0);
  {
    Matrix<Scalar,0,1> b, x;
    x = A.inverse() * b;
  }
  {
    Matrix<Scalar,Dynamic,Dynamic> b(0,1), x;
    x = A.inverse() * b;
    VERIFY_IS_EQUAL(x.rows(), 0);
    VERIFY_IS_EQUAL(x.cols(), 1);
  }
}

void test_inverse()
{
  int s = 0;
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( inverse(Matrix<double,1,1>()) );
    CALL_SUBTEST_2( inverse(Matrix2d()) );
    CALL_SUBTEST_3( inverse(Matrix3f()) );
    CALL_SUBTEST_4( inverse(Matrix4f()) );
    CALL_SUBTEST_4( inverse(Matrix<float,4,4,DontAlign>()) );
    
    s = internal::random<int>(50,320); 
    CALL_SUBTEST_5( inverse(MatrixXf(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    CALL_SUBTEST_5( inverse_zerosized<float>() );
    
    s = internal::random<int>(25,100);
    CALL_SUBTEST_6( inverse(MatrixXcd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    CALL_SUBTEST_7( inverse(Matrix4d()) );
    CALL_SUBTEST_7( inverse(Matrix<double,4,4,DontAlign>()) );

    CALL_SUBTEST_8( inverse(Matrix4cd()) );
  }
}
