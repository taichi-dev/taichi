// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "lapack_common.h"
#include <Eigen/Cholesky>

// POTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
EIGEN_LAPACK_FUNC(potrf,(char* uplo, int *n, RealScalar *pa, int *lda, int *info))
{
  *info = 0;
        if(UPLO(*uplo)==INVALID) *info = -1;
  else  if(*n<0)                 *info = -2;
  else  if(*lda<std::max(1,*n))  *info = -4;
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"POTRF", &e, 6);
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  MatrixType A(a,*n,*n,*lda);
  int ret;
  if(UPLO(*uplo)==UP) ret = int(internal::llt_inplace<Scalar, Upper>::blocked(A));
  else                ret = int(internal::llt_inplace<Scalar, Lower>::blocked(A));

  if(ret>=0)
    *info = ret+1;
  
  return 0;
}

// POTRS solves a system of linear equations A*X = B with a symmetric
// positive definite matrix A using the Cholesky factorization
// A = U**T*U or A = L*L**T computed by DPOTRF.
EIGEN_LAPACK_FUNC(potrs,(char* uplo, int *n, int *nrhs, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, int *info))
{
  *info = 0;
        if(UPLO(*uplo)==INVALID) *info = -1;
  else  if(*n<0)                 *info = -2;
  else  if(*nrhs<0)              *info = -3;
  else  if(*lda<std::max(1,*n))  *info = -5;
  else  if(*ldb<std::max(1,*n))  *info = -7;
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"POTRS", &e, 6);
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  MatrixType A(a,*n,*n,*lda);
  MatrixType B(b,*n,*nrhs,*ldb);

  if(UPLO(*uplo)==UP)
  {
    A.triangularView<Upper>().adjoint().solveInPlace(B);
    A.triangularView<Upper>().solveInPlace(B);
  }
  else
  {
    A.triangularView<Lower>().solveInPlace(B);
    A.triangularView<Lower>().adjoint().solveInPlace(B);
  }

  return 0;
}
