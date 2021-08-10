// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template<typename MatrixType> void schur(int size = MatrixType::ColsAtCompileTime)
{
  typedef typename ComplexSchur<MatrixType>::ComplexScalar ComplexScalar;
  typedef typename ComplexSchur<MatrixType>::ComplexMatrixType ComplexMatrixType;

  // Test basic functionality: T is triangular and A = U T U*
  for(int counter = 0; counter < g_repeat; ++counter) {
    MatrixType A = MatrixType::Random(size, size);
    ComplexSchur<MatrixType> schurOfA(A);
    VERIFY_IS_EQUAL(schurOfA.info(), Success);
    ComplexMatrixType U = schurOfA.matrixU();
    ComplexMatrixType T = schurOfA.matrixT();
    for(int row = 1; row < size; ++row) {
      for(int col = 0; col < row; ++col) {
        VERIFY(T(row,col) == (typename MatrixType::Scalar)0);
      }
    }
    VERIFY_IS_APPROX(A.template cast<ComplexScalar>(), U * T * U.adjoint());
  }

  // Test asserts when not initialized
  ComplexSchur<MatrixType> csUninitialized;
  VERIFY_RAISES_ASSERT(csUninitialized.matrixT());
  VERIFY_RAISES_ASSERT(csUninitialized.matrixU());
  VERIFY_RAISES_ASSERT(csUninitialized.info());
  
  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  ComplexSchur<MatrixType> cs1;
  cs1.compute(A);
  ComplexSchur<MatrixType> cs2(A);
  VERIFY_IS_EQUAL(cs1.info(), Success);
  VERIFY_IS_EQUAL(cs2.info(), Success);
  VERIFY_IS_EQUAL(cs1.matrixT(), cs2.matrixT());
  VERIFY_IS_EQUAL(cs1.matrixU(), cs2.matrixU());

  // Test maximum number of iterations
  ComplexSchur<MatrixType> cs3;
  cs3.setMaxIterations(ComplexSchur<MatrixType>::m_maxIterationsPerRow * size).compute(A);
  VERIFY_IS_EQUAL(cs3.info(), Success);
  VERIFY_IS_EQUAL(cs3.matrixT(), cs1.matrixT());
  VERIFY_IS_EQUAL(cs3.matrixU(), cs1.matrixU());
  cs3.setMaxIterations(1).compute(A);
  VERIFY_IS_EQUAL(cs3.info(), size > 1 ? NoConvergence : Success);
  VERIFY_IS_EQUAL(cs3.getMaxIterations(), 1);

  MatrixType Atriangular = A;
  Atriangular.template triangularView<StrictlyLower>().setZero(); 
  cs3.setMaxIterations(1).compute(Atriangular); // triangular matrices do not need any iterations
  VERIFY_IS_EQUAL(cs3.info(), Success);
  VERIFY_IS_EQUAL(cs3.matrixT(), Atriangular.template cast<ComplexScalar>());
  VERIFY_IS_EQUAL(cs3.matrixU(), ComplexMatrixType::Identity(size, size));

  // Test computation of only T, not U
  ComplexSchur<MatrixType> csOnlyT(A, false);
  VERIFY_IS_EQUAL(csOnlyT.info(), Success);
  VERIFY_IS_EQUAL(cs1.matrixT(), csOnlyT.matrixT());
  VERIFY_RAISES_ASSERT(csOnlyT.matrixU());

  if (size > 1 && size < 20)
  {
    // Test matrix with NaN
    A(0,0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    ComplexSchur<MatrixType> csNaN(A);
    VERIFY_IS_EQUAL(csNaN.info(), NoConvergence);
  }
}

void test_schur_complex()
{
  CALL_SUBTEST_1(( schur<Matrix4cd>() ));
  CALL_SUBTEST_2(( schur<MatrixXcf>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4)) ));
  CALL_SUBTEST_3(( schur<Matrix<std::complex<float>, 1, 1> >() ));
  CALL_SUBTEST_4(( schur<Matrix<float, 3, 3, Eigen::RowMajor> >() ));

  // Test problem size constructors
  CALL_SUBTEST_5(ComplexSchur<MatrixXf>(10));
}
