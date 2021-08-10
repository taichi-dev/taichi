// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"



template<typename MatrixType> void triangular_square(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols);
  VectorType v2 = VectorType::Random(rows);

  MatrixType m1up = m1.template triangularView<Upper>();
  MatrixType m2up = m2.template triangularView<Upper>();

  if (rows*cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

//   VERIFY_IS_APPROX(m1up.transpose() * m2, m1.upper().transpose().lower() * m2);

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.template triangularView<Upper>() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<Upper>() = m2.transpose() + m2;
  m3 = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.template triangularView<Lower>().transpose().toDenseMatrix(), m1);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<Lower>() = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.template triangularView<Lower>().toDenseMatrix(), m1);

  VERIFY_IS_APPROX(m3.template triangularView<Lower>().conjugate().toDenseMatrix(),
                   m3.conjugate().template triangularView<Lower>().toDenseMatrix());

  m1 = MatrixType::Random(rows, cols);
  for (int i=0; i<rows; ++i)
    while (numext::abs2(m1(i,i))<RealScalar(1e-1)) m1(i,i) = internal::random<Scalar>();

  Transpose<MatrixType> trm4(m4);
  // test back and forward subsitution with a vector as the rhs
  m3 = m1.template triangularView<Upper>();
  VERIFY(v2.isApprox(m3.adjoint() * (m1.adjoint().template triangularView<Lower>().solve(v2)), largerEps));
  m3 = m1.template triangularView<Lower>();
  VERIFY(v2.isApprox(m3.transpose() * (m1.transpose().template triangularView<Upper>().solve(v2)), largerEps));
  m3 = m1.template triangularView<Upper>();
  VERIFY(v2.isApprox(m3 * (m1.template triangularView<Upper>().solve(v2)), largerEps));
  m3 = m1.template triangularView<Lower>();
  VERIFY(v2.isApprox(m3.conjugate() * (m1.conjugate().template triangularView<Lower>().solve(v2)), largerEps));

  // test back and forward substitution with a matrix as the rhs
  m3 = m1.template triangularView<Upper>();
  VERIFY(m2.isApprox(m3.adjoint() * (m1.adjoint().template triangularView<Lower>().solve(m2)), largerEps));
  m3 = m1.template triangularView<Lower>();
  VERIFY(m2.isApprox(m3.transpose() * (m1.transpose().template triangularView<Upper>().solve(m2)), largerEps));
  m3 = m1.template triangularView<Upper>();
  VERIFY(m2.isApprox(m3 * (m1.template triangularView<Upper>().solve(m2)), largerEps));
  m3 = m1.template triangularView<Lower>();
  VERIFY(m2.isApprox(m3.conjugate() * (m1.conjugate().template triangularView<Lower>().solve(m2)), largerEps));

  // check M * inv(L) using in place API
  m4 = m3;
  m1.transpose().template triangularView<Eigen::Upper>().solveInPlace(trm4);
  VERIFY_IS_APPROX(m4 * m1.template triangularView<Eigen::Lower>(), m3);

  // check M * inv(U) using in place API
  m3 = m1.template triangularView<Upper>();
  m4 = m3;
  m3.transpose().template triangularView<Eigen::Lower>().solveInPlace(trm4);
  VERIFY_IS_APPROX(m4 * m1.template triangularView<Eigen::Upper>(), m3);

  // check solve with unit diagonal
  m3 = m1.template triangularView<UnitUpper>();
  VERIFY(m2.isApprox(m3 * (m1.template triangularView<UnitUpper>().solve(m2)), largerEps));

//   VERIFY((  m1.template triangularView<Upper>()
//           * m2.template triangularView<Upper>()).isUpperTriangular());

  // test swap
  m1.setOnes();
  m2.setZero();
  m2.template triangularView<Upper>().swap(m1);
  m3.setZero();
  m3.template triangularView<Upper>().setOnes();
  VERIFY_IS_APPROX(m2,m3);
  
  m1.setRandom();
  m3 = m1.template triangularView<Upper>();
  Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic> m5(cols, internal::random<int>(1,20));  m5.setRandom();
  Matrix<Scalar, Dynamic, MatrixType::RowsAtCompileTime> m6(internal::random<int>(1,20), rows);  m6.setRandom();
  VERIFY_IS_APPROX(m1.template triangularView<Upper>() * m5, m3*m5);
  VERIFY_IS_APPROX(m6*m1.template triangularView<Upper>(), m6*m3);

  m1up = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(m1.template selfadjointView<Upper>().template triangularView<Upper>().toDenseMatrix(), m1up);
  VERIFY_IS_APPROX(m1up.template selfadjointView<Upper>().template triangularView<Upper>().toDenseMatrix(), m1up);
  VERIFY_IS_APPROX(m1.template selfadjointView<Upper>().template triangularView<Lower>().toDenseMatrix(), m1up.adjoint());
  VERIFY_IS_APPROX(m1up.template selfadjointView<Upper>().template triangularView<Lower>().toDenseMatrix(), m1up.adjoint());

  VERIFY_IS_APPROX(m1.template selfadjointView<Upper>().diagonal(), m1.diagonal());

}


template<typename MatrixType> void triangular_rect(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  enum { Rows =  MatrixType::RowsAtCompileTime, Cols =  MatrixType::ColsAtCompileTime };

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols);

  MatrixType m1up = m1.template triangularView<Upper>();
  MatrixType m2up = m2.template triangularView<Upper>();

  if (rows>1 && cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.template triangularView<Upper>() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<Upper>() = 3 * m2;
  m3 = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<Upper>().toDenseMatrix(), m1);


  m1.setZero();
  m1.template triangularView<Lower>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<Lower>().toDenseMatrix(), m1);

  m1.setZero();
  m1.template triangularView<StrictlyUpper>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<StrictlyUpper>().toDenseMatrix(), m1);


  m1.setZero();
  m1.template triangularView<StrictlyLower>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<StrictlyLower>().toDenseMatrix(), m1);
  m1.setRandom();
  m2 = m1.template triangularView<Upper>();
  VERIFY(m2.isUpperTriangular());
  VERIFY(!m2.isLowerTriangular());
  m2 = m1.template triangularView<StrictlyUpper>();
  VERIFY(m2.isUpperTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<UnitUpper>();
  VERIFY(m2.isUpperTriangular());
  m2.diagonal().array() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<Lower>();
  VERIFY(m2.isLowerTriangular());
  VERIFY(!m2.isUpperTriangular());
  m2 = m1.template triangularView<StrictlyLower>();
  VERIFY(m2.isLowerTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<UnitLower>();
  VERIFY(m2.isLowerTriangular());
  m2.diagonal().array() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  // test swap
  m1.setOnes();
  m2.setZero();
  m2.template triangularView<Upper>().swap(m1);
  m3.setZero();
  m3.template triangularView<Upper>().setOnes();
  VERIFY_IS_APPROX(m2,m3);
}

void bug_159()
{
  Matrix3d m = Matrix3d::Random().triangularView<Lower>();
  EIGEN_UNUSED_VARIABLE(m)
}

void test_triangular()
{
  int maxsize = (std::min)(EIGEN_TEST_MAX_SIZE,20);
  for(int i = 0; i < g_repeat ; i++)
  {
    int r = internal::random<int>(2,maxsize); TEST_SET_BUT_UNUSED_VARIABLE(r)
    int c = internal::random<int>(2,maxsize); TEST_SET_BUT_UNUSED_VARIABLE(c)

    CALL_SUBTEST_1( triangular_square(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( triangular_square(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( triangular_square(Matrix3d()) );
    CALL_SUBTEST_4( triangular_square(Matrix<std::complex<float>,8, 8>()) );
    CALL_SUBTEST_5( triangular_square(MatrixXcd(r,r)) );
    CALL_SUBTEST_6( triangular_square(Matrix<float,Dynamic,Dynamic,RowMajor>(r, r)) );

    CALL_SUBTEST_7( triangular_rect(Matrix<float, 4, 5>()) );
    CALL_SUBTEST_8( triangular_rect(Matrix<double, 6, 2>()) );
    CALL_SUBTEST_9( triangular_rect(MatrixXcf(r, c)) );
    CALL_SUBTEST_5( triangular_rect(MatrixXcd(r, c)) );
    CALL_SUBTEST_6( triangular_rect(Matrix<float,Dynamic,Dynamic,RowMajor>(r, c)) );
  }
  
  CALL_SUBTEST_1( bug_159() );
}
