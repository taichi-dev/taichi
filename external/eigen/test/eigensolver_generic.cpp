// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template<typename MatrixType> void eigensolver(const MatrixType& m)
{
  /* this test covers the following files:
     EigenSolver.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, 1> RealVectorType;
  typedef typename std::complex<typename NumTraits<typename MatrixType::Scalar>::Real> Complex;

  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType a1 = MatrixType::Random(rows,cols);
  MatrixType symmA =  a.adjoint() * a + a1.adjoint() * a1;

  EigenSolver<MatrixType> ei0(symmA);
  VERIFY_IS_EQUAL(ei0.info(), Success);
  VERIFY_IS_APPROX(symmA * ei0.pseudoEigenvectors(), ei0.pseudoEigenvectors() * ei0.pseudoEigenvalueMatrix());
  VERIFY_IS_APPROX((symmA.template cast<Complex>()) * (ei0.pseudoEigenvectors().template cast<Complex>()),
    (ei0.pseudoEigenvectors().template cast<Complex>()) * (ei0.eigenvalues().asDiagonal()));

  EigenSolver<MatrixType> ei1(a);
  VERIFY_IS_EQUAL(ei1.info(), Success);
  VERIFY_IS_APPROX(a * ei1.pseudoEigenvectors(), ei1.pseudoEigenvectors() * ei1.pseudoEigenvalueMatrix());
  VERIFY_IS_APPROX(a.template cast<Complex>() * ei1.eigenvectors(),
                   ei1.eigenvectors() * ei1.eigenvalues().asDiagonal());
  VERIFY_IS_APPROX(ei1.eigenvectors().colwise().norm(), RealVectorType::Ones(rows).transpose());
  VERIFY_IS_APPROX(a.eigenvalues(), ei1.eigenvalues());

  EigenSolver<MatrixType> ei2;
  ei2.setMaxIterations(RealSchur<MatrixType>::m_maxIterationsPerRow * rows).compute(a);
  VERIFY_IS_EQUAL(ei2.info(), Success);
  VERIFY_IS_EQUAL(ei2.eigenvectors(), ei1.eigenvectors());
  VERIFY_IS_EQUAL(ei2.eigenvalues(), ei1.eigenvalues());
  if (rows > 2) {
    ei2.setMaxIterations(1).compute(a);
    VERIFY_IS_EQUAL(ei2.info(), NoConvergence);
    VERIFY_IS_EQUAL(ei2.getMaxIterations(), 1);
  }

  EigenSolver<MatrixType> eiNoEivecs(a, false);
  VERIFY_IS_EQUAL(eiNoEivecs.info(), Success);
  VERIFY_IS_APPROX(ei1.eigenvalues(), eiNoEivecs.eigenvalues());
  VERIFY_IS_APPROX(ei1.pseudoEigenvalueMatrix(), eiNoEivecs.pseudoEigenvalueMatrix());

  MatrixType id = MatrixType::Identity(rows, cols);
  VERIFY_IS_APPROX(id.operatorNorm(), RealScalar(1));

  if (rows > 2 && rows < 20)
  {
    // Test matrix with NaN
    a(0,0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    EigenSolver<MatrixType> eiNaN(a);
    VERIFY_IS_NOT_EQUAL(eiNaN.info(), Success);
  }

  // regression test for bug 1098
  {
    EigenSolver<MatrixType> eig(a.adjoint() * a);
    eig.compute(a.adjoint() * a);
  }

  // regression test for bug 478
  {
    a.setZero();
    EigenSolver<MatrixType> ei3(a);
    VERIFY_IS_EQUAL(ei3.info(), Success);
    VERIFY_IS_MUCH_SMALLER_THAN(ei3.eigenvalues().norm(),RealScalar(1));
    VERIFY((ei3.eigenvectors().transpose()*ei3.eigenvectors().transpose()).eval().isIdentity());
  }
}

template<typename MatrixType> void eigensolver_verify_assert(const MatrixType& m)
{
  EigenSolver<MatrixType> eig;
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvalueMatrix());
  VERIFY_RAISES_ASSERT(eig.eigenvalues());

  MatrixType a = MatrixType::Random(m.rows(),m.cols());
  eig.compute(a, false);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.pseudoEigenvectors());
}

void test_eigensolver_generic()
{
  int s = 0;
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( eigensolver(Matrix4f()) );
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
    CALL_SUBTEST_2( eigensolver(MatrixXd(s,s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_2( eigensolver(MatrixXd(1,1)) );
    CALL_SUBTEST_2( eigensolver(MatrixXd(2,2)) );
    CALL_SUBTEST_3( eigensolver(Matrix<double,1,1>()) );
    CALL_SUBTEST_4( eigensolver(Matrix2d()) );
  }

  CALL_SUBTEST_1( eigensolver_verify_assert(Matrix4f()) );
  s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
  CALL_SUBTEST_2( eigensolver_verify_assert(MatrixXd(s,s)) );
  CALL_SUBTEST_3( eigensolver_verify_assert(Matrix<double,1,1>()) );
  CALL_SUBTEST_4( eigensolver_verify_assert(Matrix2d()) );

  // Test problem size constructors
  CALL_SUBTEST_5(EigenSolver<MatrixXf> tmp(s));

  // regression test for bug 410
  CALL_SUBTEST_2(
  {
     MatrixXd A(1,1);
     A(0,0) = std::sqrt(-1.); // is Not-a-Number
     Eigen::EigenSolver<MatrixXd> solver(A);
     VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
  }
  );
  
#ifdef EIGEN_TEST_PART_2
  {
    // regression test for bug 793
    MatrixXd a(3,3);
    a << 0,  0,  1,
        1,  1, 1,
        1, 1e+200,  1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    double scale = 1e-200; // scale to avoid overflow during the comparisons
    VERIFY_IS_APPROX(a * eig.pseudoEigenvectors()*scale, eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix()*scale);
    VERIFY_IS_APPROX(a * eig.eigenvectors()*scale, eig.eigenvectors() * eig.eigenvalues().asDiagonal()*scale);
  }
  {
    // check a case where all eigenvalues are null.
    MatrixXd a(2,2);
    a << 1,  1,
        -1, -1;
    Eigen::EigenSolver<MatrixXd> eig(a);
    VERIFY_IS_APPROX(eig.pseudoEigenvectors().squaredNorm(), 2.);
    VERIFY_IS_APPROX((a * eig.pseudoEigenvectors()).norm()+1., 1.);
    VERIFY_IS_APPROX((eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix()).norm()+1., 1.);
    VERIFY_IS_APPROX((a * eig.eigenvectors()).norm()+1., 1.);
    VERIFY_IS_APPROX((eig.eigenvectors() * eig.eigenvalues().asDiagonal()).norm()+1., 1.);
  }
#endif
  
  TEST_SET_BUT_UNUSED_VARIABLE(s)
}
