// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"

/**  ZHEMV  performs the matrix-vector  operation
  *
  *     y := alpha*A*x + beta*y,
  *
  *  where alpha and beta are scalars, x and y are n element vectors and
  *  A is an n by n hermitian matrix.
  */
int EIGEN_BLAS_FUNC(hemv)(const char *uplo, const int *n, const RealScalar *palpha, const RealScalar *pa, const int *lda,
                          const RealScalar *px, const int *incx, const RealScalar *pbeta, RealScalar *py, const int *incy)
{
  typedef void (*functype)(int, const Scalar*, int, const Scalar*, Scalar*, Scalar);
  static const functype func[2] = {
    // array index: UP
    (internal::selfadjoint_matrix_vector_product<Scalar,int,ColMajor,Upper,false,false>::run),
    // array index: LO
    (internal::selfadjoint_matrix_vector_product<Scalar,int,ColMajor,Lower,false,false>::run),
  };

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* x = reinterpret_cast<const Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar alpha  = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta   = *reinterpret_cast<const Scalar*>(pbeta);

  // check arguments
  int info = 0;
  if(UPLO(*uplo)==INVALID)        info = 1;
  else if(*n<0)                   info = 2;
  else if(*lda<std::max(1,*n))    info = 5;
  else if(*incx==0)               info = 7;
  else if(*incy==0)               info = 10;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HEMV ",&info,6);

  if(*n==0)
    return 1;

  const Scalar* actual_x = get_compact_vector(x,*n,*incx);
  Scalar* actual_y = get_compact_vector(y,*n,*incy);

  if(beta!=Scalar(1))
  {
    if(beta==Scalar(0)) make_vector(actual_y, *n).setZero();
    else                make_vector(actual_y, *n) *= beta;
  }

  if(alpha!=Scalar(0))
  {
    int code = UPLO(*uplo);
    if(code>=2 || func[code]==0)
      return 0;

    func[code](*n, a, *lda, actual_x, actual_y, alpha);
  }

  if(actual_x!=x) delete[] actual_x;
  if(actual_y!=y) delete[] copy_back(actual_y,y,*n,*incy);

  return 1;
}

/**  ZHBMV  performs the matrix-vector  operation
  *
  *     y := alpha*A*x + beta*y,
  *
  *  where alpha and beta are scalars, x and y are n element vectors and
  *  A is an n by n hermitian band matrix, with k super-diagonals.
  */
// int EIGEN_BLAS_FUNC(hbmv)(char *uplo, int *n, int *k, RealScalar *alpha, RealScalar *a, int *lda,
//                           RealScalar *x, int *incx, RealScalar *beta, RealScalar *y, int *incy)
// {
//   return 1;
// }

/**  ZHPMV  performs the matrix-vector operation
  *
  *     y := alpha*A*x + beta*y,
  *
  *  where alpha and beta are scalars, x and y are n element vectors and
  *  A is an n by n hermitian matrix, supplied in packed form.
  */
// int EIGEN_BLAS_FUNC(hpmv)(char *uplo, int *n, RealScalar *alpha, RealScalar *ap, RealScalar *x, int *incx, RealScalar *beta, RealScalar *y, int *incy)
// {
//   return 1;
// }

/**  ZHPR    performs the hermitian rank 1 operation
  *
  *     A := alpha*x*conjg( x' ) + A,
  *
  *  where alpha is a real scalar, x is an n element vector and A is an
  *  n by n hermitian matrix, supplied in packed form.
  */
int EIGEN_BLAS_FUNC(hpr)(char *uplo, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *pap)
{
  typedef void (*functype)(int, Scalar*, const Scalar*, RealScalar);
  static const functype func[2] = {
    // array index: UP
    (internal::selfadjoint_packed_rank1_update<Scalar,int,ColMajor,Upper,false,Conj>::run),
    // array index: LO
    (internal::selfadjoint_packed_rank1_update<Scalar,int,ColMajor,Lower,false,Conj>::run),
  };

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* ap = reinterpret_cast<Scalar*>(pap);
  RealScalar alpha = *palpha;

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HPR  ",&info,6);

  if(alpha==Scalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x, *n, *incx);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, ap, x_cpy, alpha);

  if(x_cpy!=x)  delete[] x_cpy;

  return 1;
}

/**  ZHPR2  performs the hermitian rank 2 operation
  *
  *     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A,
  *
  *  where alpha is a scalar, x and y are n element vectors and A is an
  *  n by n hermitian matrix, supplied in packed form.
  */
int EIGEN_BLAS_FUNC(hpr2)(char *uplo, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pap)
{
  typedef void (*functype)(int, Scalar*, const Scalar*, const Scalar*, Scalar);
  static const functype func[2] = {
    // array index: UP
    (internal::packed_rank2_update_selector<Scalar,int,Upper>::run),
    // array index: LO
    (internal::packed_rank2_update_selector<Scalar,int,Lower>::run),
  };

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar* ap = reinterpret_cast<Scalar*>(pap);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  else if(*incy==0)                                                   info = 7;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HPR2 ",&info,6);

  if(alpha==Scalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x, *n, *incx);
  Scalar* y_cpy = get_compact_vector(y, *n, *incy);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, ap, x_cpy, y_cpy, alpha);

  if(x_cpy!=x)  delete[] x_cpy;
  if(y_cpy!=y)  delete[] y_cpy;

  return 1;
}

/**  ZHER   performs the hermitian rank 1 operation
  *
  *     A := alpha*x*conjg( x' ) + A,
  *
  *  where alpha is a real scalar, x is an n element vector and A is an
  *  n by n hermitian matrix.
  */
int EIGEN_BLAS_FUNC(her)(char *uplo, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *pa, int *lda)
{
  typedef void (*functype)(int, Scalar*, int, const Scalar*, const Scalar*, const Scalar&);
  static const functype func[2] = {
    // array index: UP
    (selfadjoint_rank1_update<Scalar,int,ColMajor,Upper,false,Conj>::run),
    // array index: LO
    (selfadjoint_rank1_update<Scalar,int,ColMajor,Lower,false,Conj>::run),
  };

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  RealScalar alpha = *reinterpret_cast<RealScalar*>(palpha);

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  else if(*lda<std::max(1,*n))                                        info = 7;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HER  ",&info,6);

  if(alpha==RealScalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x, *n, *incx);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, a, *lda, x_cpy, x_cpy, alpha);

  matrix(a,*n,*n,*lda).diagonal().imag().setZero();

  if(x_cpy!=x)  delete[] x_cpy;

  return 1;
}

/**  ZHER2  performs the hermitian rank 2 operation
  *
  *     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A,
  *
  *  where alpha is a scalar, x and y are n element vectors and A is an n
  *  by n hermitian matrix.
  */
int EIGEN_BLAS_FUNC(her2)(char *uplo, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pa, int *lda)
{
  typedef void (*functype)(int, Scalar*, int, const Scalar*, const Scalar*, Scalar);
  static const functype func[2] = {
    // array index: UP
    (internal::rank2_update_selector<Scalar,int,Upper>::run),
    // array index: LO
    (internal::rank2_update_selector<Scalar,int,Lower>::run),
  };

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  else if(*incy==0)                                                   info = 7;
  else if(*lda<std::max(1,*n))                                        info = 9;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HER2 ",&info,6);

  if(alpha==Scalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x, *n, *incx);
  Scalar* y_cpy = get_compact_vector(y, *n, *incy);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, a, *lda, x_cpy, y_cpy, alpha);

  matrix(a,*n,*n,*lda).diagonal().imag().setZero();

  if(x_cpy!=x)  delete[] x_cpy;
  if(y_cpy!=y)  delete[] y_cpy;

  return 1;
}

/**  ZGERU  performs the rank 1 operation
  *
  *     A := alpha*x*y' + A,
  *
  *  where alpha is a scalar, x is an m element vector, y is an n element
  *  vector and A is an m by n matrix.
  */
int EIGEN_BLAS_FUNC(geru)(int *m, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pa, int *lda)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int info = 0;
       if(*m<0)                                                       info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  else if(*incy==0)                                                   info = 7;
  else if(*lda<std::max(1,*m))                                        info = 9;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"GERU ",&info,6);

  if(alpha==Scalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x,*m,*incx);
  Scalar* y_cpy = get_compact_vector(y,*n,*incy);

  internal::general_rank1_update<Scalar,int,ColMajor,false,false>::run(*m, *n, a, *lda, x_cpy, y_cpy, alpha);

  if(x_cpy!=x)  delete[] x_cpy;
  if(y_cpy!=y)  delete[] y_cpy;

  return 1;
}

/**  ZGERC  performs the rank 1 operation
  *
  *     A := alpha*x*conjg( y' ) + A,
  *
  *  where alpha is a scalar, x is an m element vector, y is an n element
  *  vector and A is an m by n matrix.
  */
int EIGEN_BLAS_FUNC(gerc)(int *m, int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pa, int *lda)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int info = 0;
       if(*m<0)                                                       info = 1;
  else if(*n<0)                                                       info = 2;
  else if(*incx==0)                                                   info = 5;
  else if(*incy==0)                                                   info = 7;
  else if(*lda<std::max(1,*m))                                        info = 9;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"GERC ",&info,6);

  if(alpha==Scalar(0))
    return 1;

  Scalar* x_cpy = get_compact_vector(x,*m,*incx);
  Scalar* y_cpy = get_compact_vector(y,*n,*incy);

  internal::general_rank1_update<Scalar,int,ColMajor,false,Conj>::run(*m, *n, a, *lda, x_cpy, y_cpy, alpha);

  if(x_cpy!=x)  delete[] x_cpy;
  if(y_cpy!=y)  delete[] y_cpy;

  return 1;
}
