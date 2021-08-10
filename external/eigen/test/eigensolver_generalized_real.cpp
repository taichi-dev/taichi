// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

template<typename MatrixType> void generalized_eigensolver_real(const MatrixType& m)
{
  /* this test covers the following files:
     GeneralizedEigenSolver.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef std::complex<Scalar> ComplexScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType b = MatrixType::Random(rows,cols);
  MatrixType a1 = MatrixType::Random(rows,cols);
  MatrixType b1 = MatrixType::Random(rows,cols);
  MatrixType spdA =  a.adjoint() * a + a1.adjoint() * a1;
  MatrixType spdB =  b.adjoint() * b + b1.adjoint() * b1;

  // lets compare to GeneralizedSelfAdjointEigenSolver
  {
    GeneralizedSelfAdjointEigenSolver<MatrixType> symmEig(spdA, spdB);
    GeneralizedEigenSolver<MatrixType> eig(spdA, spdB);

    VERIFY_IS_EQUAL(eig.eigenvalues().imag().cwiseAbs().maxCoeff(), 0);

    VectorType realEigenvalues = eig.eigenvalues().real();
    std::sort(realEigenvalues.data(), realEigenvalues.data()+realEigenvalues.size());
    VERIFY_IS_APPROX(realEigenvalues, symmEig.eigenvalues());

    // check eigenvectors
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType D = eig.eigenvalues().asDiagonal();
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType V = eig.eigenvectors();
    VERIFY_IS_APPROX(spdA*V, spdB*V*D);
  }

  // non symmetric case:
  {
    GeneralizedEigenSolver<MatrixType> eig(rows);
    // TODO enable full-prealocation of required memory, this probably requires an in-place mode for HessenbergDecomposition
    //Eigen::internal::set_is_malloc_allowed(false);
    eig.compute(a,b);
    //Eigen::internal::set_is_malloc_allowed(true);
    for(Index k=0; k<cols; ++k)
    {
      Matrix<ComplexScalar,Dynamic,Dynamic> tmp = (eig.betas()(k)*a).template cast<ComplexScalar>() - eig.alphas()(k)*b;
      if(tmp.size()>1 && tmp.norm()>(std::numeric_limits<Scalar>::min)())
        tmp /= tmp.norm();
      VERIFY_IS_MUCH_SMALLER_THAN( std::abs(tmp.determinant()), Scalar(1) );
    }
    // check eigenvectors
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType D = eig.eigenvalues().asDiagonal();
    typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType V = eig.eigenvectors();
    VERIFY_IS_APPROX(a*V, b*V*D);
  }

  // regression test for bug 1098
  {
    GeneralizedSelfAdjointEigenSolver<MatrixType> eig1(a.adjoint() * a,b.adjoint() * b);
    eig1.compute(a.adjoint() * a,b.adjoint() * b);
    GeneralizedEigenSolver<MatrixType> eig2(a.adjoint() * a,b.adjoint() * b);
    eig2.compute(a.adjoint() * a,b.adjoint() * b);
  }

  // check without eigenvectors
  {
    GeneralizedEigenSolver<MatrixType> eig1(spdA, spdB, true);
    GeneralizedEigenSolver<MatrixType> eig2(spdA, spdB, false);
    VERIFY_IS_APPROX(eig1.eigenvalues(), eig2.eigenvalues());
  }
}

void test_eigensolver_generalized_real()
{
  for(int i = 0; i < g_repeat; i++) {
    int s = 0;
    CALL_SUBTEST_1( generalized_eigensolver_real(Matrix4f()) );
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
    CALL_SUBTEST_2( generalized_eigensolver_real(MatrixXd(s,s)) );

    // some trivial but implementation-wise special cases
    CALL_SUBTEST_2( generalized_eigensolver_real(MatrixXd(1,1)) );
    CALL_SUBTEST_2( generalized_eigensolver_real(MatrixXd(2,2)) );
    CALL_SUBTEST_3( generalized_eigensolver_real(Matrix<double,1,1>()) );
    CALL_SUBTEST_4( generalized_eigensolver_real(Matrix2d()) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
