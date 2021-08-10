// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"
#include <Eigen/LU>

// computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges
EIGEN_LAPACK_FUNC(getrf,(int *m, int *n, RealScalar *pa, int *lda, int *ipiv, int *info))
{
  *info = 0;
        if(*m<0)                  *info = -1;
  else  if(*n<0)                  *info = -2;
  else  if(*lda<std::max(1,*m))   *info = -4;
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"GETRF", &e, 6);
  }

  if(*m==0 || *n==0)
    return 0;

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  int nb_transpositions;
  int ret = int(Eigen::internal::partial_lu_impl<Scalar,ColMajor,int>
                     ::blocked_lu(*m, *n, a, *lda, ipiv, nb_transpositions));

  for(int i=0; i<std::min(*m,*n); ++i)
    ipiv[i]++;

  if(ret>=0)
    *info = ret+1;

  return 0;
}

//GETRS solves a system of linear equations
//    A * X = B  or  A' * X = B
//  with a general N-by-N matrix A using the LU factorization computed  by GETRF
EIGEN_LAPACK_FUNC(getrs,(char *trans, int *n, int *nrhs, RealScalar *pa, int *lda, int *ipiv, RealScalar *pb, int *ldb, int *info))
{
  *info = 0;
        if(OP(*trans)==INVALID)  *info = -1;
  else  if(*n<0)                 *info = -2;
  else  if(*nrhs<0)              *info = -3;
  else  if(*lda<std::max(1,*n))  *info = -5;
  else  if(*ldb<std::max(1,*n))  *info = -8;
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"GETRS", &e, 6);
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  MatrixType lu(a,*n,*n,*lda);
  MatrixType B(b,*n,*nrhs,*ldb);

  for(int i=0; i<*n; ++i)
    ipiv[i]--;
  if(OP(*trans)==NOTR)
  {
    B = PivotsType(ipiv,*n) * B;
    lu.triangularView<UnitLower>().solveInPlace(B);
    lu.triangularView<Upper>().solveInPlace(B);
  }
  else if(OP(*trans)==TR)
  {
    lu.triangularView<Upper>().transpose().solveInPlace(B);
    lu.triangularView<UnitLower>().transpose().solveInPlace(B);
    B = PivotsType(ipiv,*n).transpose() * B;
  }
  else if(OP(*trans)==ADJ)
  {
    lu.triangularView<Upper>().adjoint().solveInPlace(B);
    lu.triangularView<UnitLower>().adjoint().solveInPlace(B);
    B = PivotsType(ipiv,*n).transpose() * B;
  }
  for(int i=0; i<*n; ++i)
    ipiv[i]++;

  return 0;
}
