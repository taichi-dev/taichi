// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "product.h"

template<typename T>
void test_aliasing()
{
  int rows = internal::random<int>(1,12);
  int cols = internal::random<int>(1,12);
  typedef Matrix<T,Dynamic,Dynamic> MatrixType;
  typedef Matrix<T,Dynamic,1> VectorType;
  VectorType x(cols); x.setRandom();
  VectorType z(x);
  VectorType y(rows); y.setZero();
  MatrixType A(rows,cols); A.setRandom();
  // CwiseBinaryOp
  VERIFY_IS_APPROX(x = y + A*x, A*z);     // OK because "y + A*x" is marked as "assume-aliasing"
  x = z;
  // CwiseUnaryOp
  VERIFY_IS_APPROX(x = T(1.)*(A*x), A*z); // OK because 1*(A*x) is replaced by (1*A*x) which is a Product<> expression
  x = z;
  // VERIFY_IS_APPROX(x = y-A*x, -A*z);   // Not OK in 3.3 because x is resized before A*x gets evaluated
  x = z;
}

void test_product_large()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( product(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( product(MatrixXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( product(MatrixXd(internal::random<int>(1,10), internal::random<int>(1,10))) );

    CALL_SUBTEST_3( product(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_4( product(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_5( product(Matrix<float,Dynamic,Dynamic,RowMajor>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );

    CALL_SUBTEST_1( test_aliasing<float>() );
  }

#if defined EIGEN_TEST_PART_6
  {
    // test a specific issue in DiagonalProduct
    int N = 1000000;
    VectorXf v = VectorXf::Ones(N);
    MatrixXf m = MatrixXf::Ones(N,3);
    m = (v+v).asDiagonal() * m;
    VERIFY_IS_APPROX(m, MatrixXf::Constant(N,3,2));
  }

  {
    // test deferred resizing in Matrix::operator=
    MatrixXf a = MatrixXf::Random(10,4), b = MatrixXf::Random(4,10), c = a;
    VERIFY_IS_APPROX((a = a * b), (c * b).eval());
  }

  {
    // check the functions to setup blocking sizes compile and do not segfault
    // FIXME check they do what they are supposed to do !!
    std::ptrdiff_t l1 = internal::random<int>(10000,20000);
    std::ptrdiff_t l2 = internal::random<int>(100000,200000);
    std::ptrdiff_t l3 = internal::random<int>(1000000,2000000);
    setCpuCacheSizes(l1,l2,l3);
    VERIFY(l1==l1CacheSize());
    VERIFY(l2==l2CacheSize());
    std::ptrdiff_t k1 = internal::random<int>(10,100)*16;
    std::ptrdiff_t m1 = internal::random<int>(10,100)*16;
    std::ptrdiff_t n1 = internal::random<int>(10,100)*16;
    // only makes sure it compiles fine
    internal::computeProductBlockingSizes<float,float,std::ptrdiff_t>(k1,m1,n1,1);
  }

  {
    // test regression in row-vector by matrix (bad Map type)
    MatrixXf mat1(10,32); mat1.setRandom();
    MatrixXf mat2(32,32); mat2.setRandom();
    MatrixXf r1 = mat1.row(2)*mat2.transpose();
    VERIFY_IS_APPROX(r1, (mat1.row(2)*mat2.transpose()).eval());

    MatrixXf r2 = mat1.row(2)*mat2;
    VERIFY_IS_APPROX(r2, (mat1.row(2)*mat2).eval());
  }

  {
    Eigen::MatrixXd A(10,10), B, C;
    A.setRandom();
    C = A;
    for(int k=0; k<79; ++k)
      C = C * A;
    B.noalias() = (((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)) * ((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)))
                * (((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)) * ((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A))*((A*A)*(A*A)));
    VERIFY_IS_APPROX(B,C);
  }
#endif

  // Regression test for bug 714:
#if defined EIGEN_HAS_OPENMP
  omp_set_dynamic(1);
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_6( product(Matrix<float,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
#endif
}
