// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"

template<typename MatrixType> struct Wrapper
{
  MatrixType m_mat;
  inline Wrapper(const MatrixType &x) : m_mat(x) {}
  inline operator const MatrixType& () const { return m_mat; }
  inline operator MatrixType& () { return m_mat; }
};

enum my_sizes { M = 12, N = 7};

template<typename MatrixType> void ctor_init1(const MatrixType& m)
{
  // Check logic in PlainObjectBase::_init1
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m0 = MatrixType::Random(rows,cols);

  VERIFY_EVALUATION_COUNT( MatrixType m1(m0), 1);
  VERIFY_EVALUATION_COUNT( MatrixType m2(m0+m0), 1);
  VERIFY_EVALUATION_COUNT( MatrixType m2(m0.block(0,0,rows,cols)) , 1);

  Wrapper<MatrixType> wrapper(m0);
  VERIFY_EVALUATION_COUNT( MatrixType m3(wrapper) , 1);
}


void test_constructor()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( ctor_init1(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( ctor_init1(Matrix4d()) );
    CALL_SUBTEST_1( ctor_init1(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_1( ctor_init1(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  {
    Matrix<Index,1,1> a(123);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Matrix<Index,1,1> a(123.0);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Matrix<float,1,1> a(123);
    VERIFY_IS_EQUAL(a[0], 123.f);
  }
  {
    Array<Index,1,1> a(123);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Array<Index,1,1> a(123.0);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Array<float,1,1> a(123);
    VERIFY_IS_EQUAL(a[0], 123.f);
  }
  {
    Array<Index,3,3> a(123);
    VERIFY_IS_EQUAL(a(4), 123);
  }
  {
    Array<Index,3,3> a(123.0);
    VERIFY_IS_EQUAL(a(4), 123);
  }
  {
    Array<float,3,3> a(123);
    VERIFY_IS_EQUAL(a(4), 123.f);
  }
  {
    MatrixXi m1(M,N);
    VERIFY_IS_EQUAL(m1.rows(),M);
    VERIFY_IS_EQUAL(m1.cols(),N);
    ArrayXXi a1(M,N);
    VERIFY_IS_EQUAL(a1.rows(),M);
    VERIFY_IS_EQUAL(a1.cols(),N);
    VectorXi v1(M);
    VERIFY_IS_EQUAL(v1.size(),M);
    ArrayXi a2(M);
    VERIFY_IS_EQUAL(a2.size(),M);
  }
}
