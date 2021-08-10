// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Desire Nuentsa Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse.h"
#include <Eigen/SPQRSupport>


template<typename MatrixType,typename DenseMat>
int generate_sparse_rectangular_problem(MatrixType& A, DenseMat& dA, int maxRows = 300, int maxCols = 300)
{
  eigen_assert(maxRows >= maxCols);
  typedef typename MatrixType::Scalar Scalar;
  int rows = internal::random<int>(1,maxRows);
  int cols = internal::random<int>(1,rows);
  double density = (std::max)(8./(rows*cols), 0.01);
  
  A.resize(rows,cols);
  dA.resize(rows,cols);
  initSparse<Scalar>(density, dA, A,ForceNonZeroDiag);
  A.makeCompressed();
  return rows;
}

template<typename Scalar> void test_spqr_scalar()
{
  typedef SparseMatrix<Scalar,ColMajor> MatrixType; 
  MatrixType A;
  Matrix<Scalar,Dynamic,Dynamic> dA;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  DenseVector refX,x,b; 
  SPQR<MatrixType> solver; 
  generate_sparse_rectangular_problem(A,dA);
  
  Index m = A.rows();
  b = DenseVector::Random(m);
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse QR factorization failed\n";
    exit(0);
    return;
  }
  x = solver.solve(b);
  if (solver.info() != Success)
  {
    std::cerr << "sparse QR factorization failed\n";
    exit(0);
    return;
  }  
  //Compare with a dense solver
  refX = dA.colPivHouseholderQr().solve(b);
  VERIFY(x.isApprox(refX,test_precision<Scalar>()));
}
void test_spqr_support()
{
  CALL_SUBTEST_1(test_spqr_scalar<double>());
  CALL_SUBTEST_2(test_spqr_scalar<std::complex<double> >());
}
