// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "lapack_common.h"
#include <Eigen/SVD>

// computes the singular values/vectors a general M-by-N matrix A using divide-and-conquer
EIGEN_LAPACK_FUNC(gesdd,(char *jobz, int *m, int* n, Scalar* a, int *lda, RealScalar *s, Scalar *u, int *ldu, Scalar *vt, int *ldvt, Scalar* /*work*/, int* lwork,
                         EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar */*rwork*/) int * /*iwork*/, int *info))
{
  // TODO exploit the work buffer
  bool query_size = *lwork==-1;
  int diag_size = (std::min)(*m,*n);
  
  *info = 0;
        if(*jobz!='A' && *jobz!='S' && *jobz!='O' && *jobz!='N')  *info = -1;
  else  if(*m<0)                                                  *info = -2;
  else  if(*n<0)                                                  *info = -3;
  else  if(*lda<std::max(1,*m))                                   *info = -5;
  else  if(*lda<std::max(1,*m))                                   *info = -8;
  else  if(*ldu <1 || (*jobz=='A' && *ldu <*m)
                   || (*jobz=='O' && *m<*n && *ldu<*m))           *info = -8;
  else  if(*ldvt<1 || (*jobz=='A' && *ldvt<*n)
                   || (*jobz=='S' && *ldvt<diag_size)
                   || (*jobz=='O' && *m>=*n && *ldvt<*n))         *info = -10;
  
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"GESDD ", &e, 6);
  }
  
  if(query_size)
  {
    *lwork = 0;
    return 0;
  }
  
  if(*n==0 || *m==0)
    return 0;
  
  PlainMatrixType mat(*m,*n);
  mat = matrix(a,*m,*n,*lda);
  
  int option = *jobz=='A' ? ComputeFullU|ComputeFullV
             : *jobz=='S' ? ComputeThinU|ComputeThinV
             : *jobz=='O' ? ComputeThinU|ComputeThinV
             : 0;

  BDCSVD<PlainMatrixType> svd(mat,option);
  
  make_vector(s,diag_size) = svd.singularValues().head(diag_size);

  if(*jobz=='A')
  {
    matrix(u,*m,*m,*ldu)   = svd.matrixU();
    matrix(vt,*n,*n,*ldvt) = svd.matrixV().adjoint();
  }
  else if(*jobz=='S')
  {
    matrix(u,*m,diag_size,*ldu)   = svd.matrixU();
    matrix(vt,diag_size,*n,*ldvt) = svd.matrixV().adjoint();
  }
  else if(*jobz=='O' && *m>=*n)
  {
    matrix(a,*m,*n,*lda)   = svd.matrixU();
    matrix(vt,*n,*n,*ldvt) = svd.matrixV().adjoint();
  }
  else if(*jobz=='O')
  {
    matrix(u,*m,*m,*ldu)        = svd.matrixU();
    matrix(a,diag_size,*n,*lda) = svd.matrixV().adjoint();
  }
    
  return 0;
}

// computes the singular values/vectors a general M-by-N matrix A using two sided jacobi algorithm
EIGEN_LAPACK_FUNC(gesvd,(char *jobu, char *jobv, int *m, int* n, Scalar* a, int *lda, RealScalar *s, Scalar *u, int *ldu, Scalar *vt, int *ldvt, Scalar* /*work*/, int* lwork,
                         EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar */*rwork*/) int *info))
{
  // TODO exploit the work buffer
  bool query_size = *lwork==-1;
  int diag_size = (std::min)(*m,*n);
  
  *info = 0;
        if( *jobu!='A' && *jobu!='S' && *jobu!='O' && *jobu!='N') *info = -1;
  else  if((*jobv!='A' && *jobv!='S' && *jobv!='O' && *jobv!='N')
           || (*jobu=='O' && *jobv=='O'))                         *info = -2;
  else  if(*m<0)                                                  *info = -3;
  else  if(*n<0)                                                  *info = -4;
  else  if(*lda<std::max(1,*m))                                   *info = -6;
  else  if(*ldu <1 || ((*jobu=='A' || *jobu=='S') && *ldu<*m))    *info = -9;
  else  if(*ldvt<1 || (*jobv=='A' && *ldvt<*n)
                   || (*jobv=='S' && *ldvt<diag_size))            *info = -11;
  
  if(*info!=0)
  {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP"GESVD ", &e, 6);
  }
  
  if(query_size)
  {
    *lwork = 0;
    return 0;
  }
  
  if(*n==0 || *m==0)
    return 0;
  
  PlainMatrixType mat(*m,*n);
  mat = matrix(a,*m,*n,*lda);
  
  int option = (*jobu=='A' ? ComputeFullU : *jobu=='S' || *jobu=='O' ? ComputeThinU : 0)
             | (*jobv=='A' ? ComputeFullV : *jobv=='S' || *jobv=='O' ? ComputeThinV : 0);
  
  JacobiSVD<PlainMatrixType> svd(mat,option);
  
  make_vector(s,diag_size) = svd.singularValues().head(diag_size);
  {
        if(*jobu=='A') matrix(u,*m,*m,*ldu)           = svd.matrixU();
  else  if(*jobu=='S') matrix(u,*m,diag_size,*ldu)    = svd.matrixU();
  else  if(*jobu=='O') matrix(a,*m,diag_size,*lda)    = svd.matrixU();
  }
  {
        if(*jobv=='A') matrix(vt,*n,*n,*ldvt)         = svd.matrixV().adjoint();
  else  if(*jobv=='S') matrix(vt,diag_size,*n,*ldvt)  = svd.matrixV().adjoint();
  else  if(*jobv=='O') matrix(a,diag_size,*n,*lda)    = svd.matrixV().adjoint();
  }
  return 0;
}
