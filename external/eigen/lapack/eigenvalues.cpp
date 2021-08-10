// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "lapack_common.h"
#include <Eigen/Eigenvalues>

// computes eigen values and vectors of a general N-by-N matrix A
EIGEN_LAPACK_FUNC(syev,(char *jobz, char *uplo, int* n, Scalar* a, int *lda, Scalar* w, Scalar* /*work*/, int* lwork, int *info))
{
  // TODO exploit the work buffer
  bool query_size = *lwork==-1;
  
  *info = 0;
        if(*jobz!='N' && *jobz!='V')                    *info = -1;
  else  if(UPLO(*uplo)==INVALID)                        *info = -2;
  else  if(*n<0)                                        *info = -3;
  else  if(*lda<std::max(1,*n))                         *info = -5;
  else  if((!query_size) && *lwork<std::max(1,3**n-1))  *info = -8;
    
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"SYEV ", &e, 6);
  }
  
  if(query_size)
  {
    *lwork = 0;
    return 0;
  }
  
  if(*n==0)
    return 0;
  
  PlainMatrixType mat(*n,*n);
  if(UPLO(*uplo)==UP) mat = matrix(a,*n,*n,*lda).adjoint();
  else                mat = matrix(a,*n,*n,*lda);
  
  bool computeVectors = *jobz=='V' || *jobz=='v';
  SelfAdjointEigenSolver<PlainMatrixType> eig(mat,computeVectors?ComputeEigenvectors:EigenvaluesOnly);
  
  if(eig.info()==NoConvergence)
  {
    make_vector(w,*n).setZero();
    if(computeVectors)
      matrix(a,*n,*n,*lda).setIdentity();
    //*info = 1;
    return 0;
  }
  
  make_vector(w,*n) = eig.eigenvalues();
  if(computeVectors)
    matrix(a,*n,*n,*lda) = eig.eigenvectors();
  
  return 0;
}
