// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Eigenvalues>

template<typename Scalar,int Size> void hessenberg(int size = Size)
{
  typedef Matrix<Scalar,Size,Size> MatrixType;

  // Test basic functionality: A = U H U* and H is Hessenberg
  for(int counter = 0; counter < g_repeat; ++counter) {
    MatrixType m = MatrixType::Random(size,size);
    HessenbergDecomposition<MatrixType> hess(m);
    MatrixType Q = hess.matrixQ();
    MatrixType H = hess.matrixH();
    VERIFY_IS_APPROX(m, Q * H * Q.adjoint());
    for(int row = 2; row < size; ++row) {
      for(int col = 0; col < row-1; ++col) {
	VERIFY(H(row,col) == (typename MatrixType::Scalar)0);
      }
    }
  }

  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  HessenbergDecomposition<MatrixType> cs1;
  cs1.compute(A);
  HessenbergDecomposition<MatrixType> cs2(A);
  VERIFY_IS_EQUAL(cs1.matrixH().eval(), cs2.matrixH().eval());
  MatrixType cs1Q = cs1.matrixQ();
  MatrixType cs2Q = cs2.matrixQ();  
  VERIFY_IS_EQUAL(cs1Q, cs2Q);

  // Test assertions for when used uninitialized
  HessenbergDecomposition<MatrixType> hessUninitialized;
  VERIFY_RAISES_ASSERT( hessUninitialized.matrixH() );
  VERIFY_RAISES_ASSERT( hessUninitialized.matrixQ() );
  VERIFY_RAISES_ASSERT( hessUninitialized.householderCoefficients() );
  VERIFY_RAISES_ASSERT( hessUninitialized.packedMatrix() );

  // TODO: Add tests for packedMatrix() and householderCoefficients()
}

void test_hessenberg()
{
  CALL_SUBTEST_1(( hessenberg<std::complex<double>,1>() ));
  CALL_SUBTEST_2(( hessenberg<std::complex<double>,2>() ));
  CALL_SUBTEST_3(( hessenberg<std::complex<float>,4>() ));
  CALL_SUBTEST_4(( hessenberg<float,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
  CALL_SUBTEST_5(( hessenberg<std::complex<double>,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));

  // Test problem size constructors
  CALL_SUBTEST_6(HessenbergDecomposition<MatrixXf>(10));
}
