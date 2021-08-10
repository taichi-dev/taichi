// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse.h"

template<typename Scalar> void
initSPD(double density,
        Matrix<Scalar,Dynamic,Dynamic>& refMat,
        SparseMatrix<Scalar>& sparseMat)
{
  Matrix<Scalar,Dynamic,Dynamic> aux(refMat.rows(),refMat.cols());
  initSparse(density,refMat,sparseMat);
  refMat = refMat * refMat.adjoint();
  for (int k=0; k<2; ++k)
  {
    initSparse(density,aux,sparseMat,ForceNonZeroDiag);
    refMat += aux * aux.adjoint();
  }
  sparseMat.setZero();
  for (int j=0 ; j<sparseMat.cols(); ++j)
    for (int i=j ; i<sparseMat.rows(); ++i)
      if (refMat(i,j)!=Scalar(0))
        sparseMat.insert(i,j) = refMat(i,j);
  sparseMat.finalize();
}

template<typename Scalar> void sparse_solvers(int rows, int cols)
{
  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  // Scalar eps = 1e-6;

  DenseVector vec1 = DenseVector::Random(rows);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;

  // test triangular solver
  {
    DenseVector vec2 = vec1, vec3 = vec1;
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);

    // lower - dense
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Lower>().solve(vec2),
                     m2.template triangularView<Lower>().solve(vec3));

    // upper - dense
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Upper>().solve(vec2),
                     m2.template triangularView<Upper>().solve(vec3));
    VERIFY_IS_APPROX(refMat2.conjugate().template triangularView<Upper>().solve(vec2),
                     m2.conjugate().template triangularView<Upper>().solve(vec3));
    {
      SparseMatrix<Scalar> cm2(m2);
      //Index rows, Index cols, Index nnz, Index* outerIndexPtr, Index* innerIndexPtr, Scalar* valuePtr
      MappedSparseMatrix<Scalar> mm2(rows, cols, cm2.nonZeros(), cm2.outerIndexPtr(), cm2.innerIndexPtr(), cm2.valuePtr());
      VERIFY_IS_APPROX(refMat2.conjugate().template triangularView<Upper>().solve(vec2),
                       mm2.conjugate().template triangularView<Upper>().solve(vec3));
    }

    // lower - transpose
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.transpose().template triangularView<Upper>().solve(vec2),
                     m2.transpose().template triangularView<Upper>().solve(vec3));

    // upper - transpose
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.transpose().template triangularView<Lower>().solve(vec2),
                     m2.transpose().template triangularView<Lower>().solve(vec3));

    SparseMatrix<Scalar> matB(rows, rows);
    DenseMatrix refMatB = DenseMatrix::Zero(rows, rows);

    // lower - sparse
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular);
    initSparse<Scalar>(density, refMatB, matB);
    refMat2.template triangularView<Lower>().solveInPlace(refMatB);
    m2.template triangularView<Lower>().solveInPlace(matB);
    VERIFY_IS_APPROX(matB.toDense(), refMatB);

    // upper - sparse
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular);
    initSparse<Scalar>(density, refMatB, matB);
    refMat2.template triangularView<Upper>().solveInPlace(refMatB);
    m2.template triangularView<Upper>().solveInPlace(matB);
    VERIFY_IS_APPROX(matB, refMatB);

    // test deprecated API
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template triangularView<Lower>().solve(vec2),
                     m2.template triangularView<Lower>().solve(vec3));
  }
}

void test_sparse_solvers()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(sparse_solvers<double>(8, 8) );
    int s = internal::random<int>(1,300);
    CALL_SUBTEST_2(sparse_solvers<std::complex<double> >(s,s) );
    CALL_SUBTEST_1(sparse_solvers<double>(s,s) );
  }
}
