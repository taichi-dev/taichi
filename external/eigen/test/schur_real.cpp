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

template<typename MatrixType> void verifyIsQuasiTriangular(const MatrixType& T)
{
  const Index size = T.cols();
  typedef typename MatrixType::Scalar Scalar;

  // Check T is lower Hessenberg
  for(int row = 2; row < size; ++row) {
    for(int col = 0; col < row - 1; ++col) {
      VERIFY(T(row,col) == Scalar(0));
    }
  }

  // Check that any non-zero on the subdiagonal is followed by a zero and is
  // part of a 2x2 diagonal block with imaginary eigenvalues.
  for(int row = 1; row < size; ++row) {
    if (T(row,row-1) != Scalar(0)) {
      VERIFY(row == size-1 || T(row+1,row) == 0);
      Scalar tr = T(row-1,row-1) + T(row,row);
      Scalar det = T(row-1,row-1) * T(row,row) - T(row-1,row) * T(row,row-1);
      VERIFY(4 * det > tr * tr);
    }
  }
}

template<typename MatrixType> void schur(int size = MatrixType::ColsAtCompileTime)
{
  // Test basic functionality: T is quasi-triangular and A = U T U*
  for(int counter = 0; counter < g_repeat; ++counter) {
    MatrixType A = MatrixType::Random(size, size);
    RealSchur<MatrixType> schurOfA(A);
    VERIFY_IS_EQUAL(schurOfA.info(), Success);
    MatrixType U = schurOfA.matrixU();
    MatrixType T = schurOfA.matrixT();
    verifyIsQuasiTriangular(T);
    VERIFY_IS_APPROX(A, U * T * U.transpose());
  }

  // Test asserts when not initialized
  RealSchur<MatrixType> rsUninitialized;
  VERIFY_RAISES_ASSERT(rsUninitialized.matrixT());
  VERIFY_RAISES_ASSERT(rsUninitialized.matrixU());
  VERIFY_RAISES_ASSERT(rsUninitialized.info());
  
  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  RealSchur<MatrixType> rs1;
  rs1.compute(A);
  RealSchur<MatrixType> rs2(A);
  VERIFY_IS_EQUAL(rs1.info(), Success);
  VERIFY_IS_EQUAL(rs2.info(), Success);
  VERIFY_IS_EQUAL(rs1.matrixT(), rs2.matrixT());
  VERIFY_IS_EQUAL(rs1.matrixU(), rs2.matrixU());

  // Test maximum number of iterations
  RealSchur<MatrixType> rs3;
  rs3.setMaxIterations(RealSchur<MatrixType>::m_maxIterationsPerRow * size).compute(A);
  VERIFY_IS_EQUAL(rs3.info(), Success);
  VERIFY_IS_EQUAL(rs3.matrixT(), rs1.matrixT());
  VERIFY_IS_EQUAL(rs3.matrixU(), rs1.matrixU());
  if (size > 2) {
    rs3.setMaxIterations(1).compute(A);
    VERIFY_IS_EQUAL(rs3.info(), NoConvergence);
    VERIFY_IS_EQUAL(rs3.getMaxIterations(), 1);
  }

  MatrixType Atriangular = A;
  Atriangular.template triangularView<StrictlyLower>().setZero(); 
  rs3.setMaxIterations(1).compute(Atriangular); // triangular matrices do not need any iterations
  VERIFY_IS_EQUAL(rs3.info(), Success);
  VERIFY_IS_APPROX(rs3.matrixT(), Atriangular); // approx because of scaling...
  VERIFY_IS_EQUAL(rs3.matrixU(), MatrixType::Identity(size, size));

  // Test computation of only T, not U
  RealSchur<MatrixType> rsOnlyT(A, false);
  VERIFY_IS_EQUAL(rsOnlyT.info(), Success);
  VERIFY_IS_EQUAL(rs1.matrixT(), rsOnlyT.matrixT());
  VERIFY_RAISES_ASSERT(rsOnlyT.matrixU());

  if (size > 2 && size < 20)
  {
    // Test matrix with NaN
    A(0,0) = std::numeric_limits<typename MatrixType::Scalar>::quiet_NaN();
    RealSchur<MatrixType> rsNaN(A);
    VERIFY_IS_EQUAL(rsNaN.info(), NoConvergence);
  }
}

void test_schur_real()
{
  CALL_SUBTEST_1(( schur<Matrix4f>() ));
  CALL_SUBTEST_2(( schur<MatrixXd>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4)) ));
  CALL_SUBTEST_3(( schur<Matrix<float, 1, 1> >() ));
  CALL_SUBTEST_4(( schur<Matrix<double, 3, 3, Eigen::RowMajor> >() ));

  // Test problem size constructors
  CALL_SUBTEST_5(RealSchur<MatrixXf>(10));
}
