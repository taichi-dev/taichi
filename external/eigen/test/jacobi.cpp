// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/SVD>

template<typename MatrixType, typename JacobiScalar>
void jacobi(const MatrixType& m = MatrixType())
{
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef Matrix<JacobiScalar, 2, 1> JacobiVector;

  const MatrixType a(MatrixType::Random(rows, cols));

  JacobiVector v = JacobiVector::Random().normalized();
  JacobiScalar c = v.x(), s = v.y();
  JacobiRotation<JacobiScalar> rot(c, s);

  {
    Index p = internal::random<Index>(0, rows-1);
    Index q;
    do {
      q = internal::random<Index>(0, rows-1);
    } while (q == p);

    MatrixType b = a;
    b.applyOnTheLeft(p, q, rot);
    VERIFY_IS_APPROX(b.row(p), c * a.row(p) + numext::conj(s) * a.row(q));
    VERIFY_IS_APPROX(b.row(q), -s * a.row(p) + numext::conj(c) * a.row(q));
  }

  {
    Index p = internal::random<Index>(0, cols-1);
    Index q;
    do {
      q = internal::random<Index>(0, cols-1);
    } while (q == p);

    MatrixType b = a;
    b.applyOnTheRight(p, q, rot);
    VERIFY_IS_APPROX(b.col(p), c * a.col(p) - s * a.col(q));
    VERIFY_IS_APPROX(b.col(q), numext::conj(s) * a.col(p) + numext::conj(c) * a.col(q));
  }
}

void test_jacobi()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( jacobi<Matrix3f, float>() ));
    CALL_SUBTEST_2(( jacobi<Matrix4d, double>() ));
    CALL_SUBTEST_3(( jacobi<Matrix4cf, float>() ));
    CALL_SUBTEST_3(( jacobi<Matrix4cf, std::complex<float> >() ));

    int r = internal::random<int>(2, internal::random<int>(1,EIGEN_TEST_MAX_SIZE)/2),
        c = internal::random<int>(2, internal::random<int>(1,EIGEN_TEST_MAX_SIZE)/2);
    CALL_SUBTEST_4(( jacobi<MatrixXf, float>(MatrixXf(r,c)) ));
    CALL_SUBTEST_5(( jacobi<MatrixXcd, double>(MatrixXcd(r,c)) ));
    CALL_SUBTEST_5(( jacobi<MatrixXcd, std::complex<double> >(MatrixXcd(r,c)) ));
    // complex<float> is really important to test as it is the only way to cover conjugation issues in certain unaligned paths
    CALL_SUBTEST_6(( jacobi<MatrixXcf, float>(MatrixXcf(r,c)) ));
    CALL_SUBTEST_6(( jacobi<MatrixXcf, std::complex<float> >(MatrixXcf(r,c)) ));
    
    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);
  }
}
