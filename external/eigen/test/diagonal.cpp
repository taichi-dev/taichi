// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void diagonal(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);

  Scalar s1 = internal::random<Scalar>();

  //check diagonal()
  VERIFY_IS_APPROX(m1.diagonal(), m1.transpose().diagonal());
  m2.diagonal() = 2 * m1.diagonal();
  m2.diagonal()[0] *= 3;

  if (rows>2)
  {
    enum {
      N1 = MatrixType::RowsAtCompileTime>2 ?  2 : 0,
      N2 = MatrixType::RowsAtCompileTime>1 ? -1 : 0
    };

    // check sub/super diagonal
    if(MatrixType::SizeAtCompileTime!=Dynamic)
    {
      VERIFY(m1.template diagonal<N1>().RowsAtCompileTime == m1.diagonal(N1).size());
      VERIFY(m1.template diagonal<N2>().RowsAtCompileTime == m1.diagonal(N2).size());
    }

    m2.template diagonal<N1>() = 2 * m1.template diagonal<N1>();
    VERIFY_IS_APPROX(m2.template diagonal<N1>(), static_cast<Scalar>(2) * m1.diagonal(N1));
    m2.template diagonal<N1>()[0] *= 3;
    VERIFY_IS_APPROX(m2.template diagonal<N1>()[0], static_cast<Scalar>(6) * m1.template diagonal<N1>()[0]);


    m2.template diagonal<N2>() = 2 * m1.template diagonal<N2>();
    m2.template diagonal<N2>()[0] *= 3;
    VERIFY_IS_APPROX(m2.template diagonal<N2>()[0], static_cast<Scalar>(6) * m1.template diagonal<N2>()[0]);

    m2.diagonal(N1) = 2 * m1.diagonal(N1);
    VERIFY_IS_APPROX(m2.template diagonal<N1>(), static_cast<Scalar>(2) * m1.diagonal(N1));
    m2.diagonal(N1)[0] *= 3;
    VERIFY_IS_APPROX(m2.diagonal(N1)[0], static_cast<Scalar>(6) * m1.diagonal(N1)[0]);

    m2.diagonal(N2) = 2 * m1.diagonal(N2);
    VERIFY_IS_APPROX(m2.template diagonal<N2>(), static_cast<Scalar>(2) * m1.diagonal(N2));
    m2.diagonal(N2)[0] *= 3;
    VERIFY_IS_APPROX(m2.diagonal(N2)[0], static_cast<Scalar>(6) * m1.diagonal(N2)[0]);

    m2.diagonal(N2).x() = s1;
    VERIFY_IS_APPROX(m2.diagonal(N2).x(), s1);
    m2.diagonal(N2).coeffRef(0) = Scalar(2)*s1;
    VERIFY_IS_APPROX(m2.diagonal(N2).coeff(0), Scalar(2)*s1);
  }

  VERIFY( m1.diagonal( cols).size()==0 );
  VERIFY( m1.diagonal(-rows).size()==0 );
}

template<typename MatrixType> void diagonal_assert(const MatrixType& m) {
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  if (rows>=2 && cols>=2)
  {
    VERIFY_RAISES_ASSERT( m1 += m1.diagonal() );
    VERIFY_RAISES_ASSERT( m1 -= m1.diagonal() );
    VERIFY_RAISES_ASSERT( m1.array() *= m1.diagonal().array() );
    VERIFY_RAISES_ASSERT( m1.array() /= m1.diagonal().array() );
  }

  VERIFY_RAISES_ASSERT( m1.diagonal(cols+1) );
  VERIFY_RAISES_ASSERT( m1.diagonal(-(rows+1)) );
}

void test_diagonal()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( diagonal(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( diagonal(Matrix<float, 4, 9>()) );
    CALL_SUBTEST_1( diagonal(Matrix<float, 7, 3>()) );
    CALL_SUBTEST_2( diagonal(Matrix4d()) );
    CALL_SUBTEST_2( diagonal(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( diagonal(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( diagonal(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_1( diagonal(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_1( diagonal(Matrix<float,Dynamic,4>(3, 4)) );
    CALL_SUBTEST_1( diagonal_assert(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
}
