// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"

int EIGEN_BLAS_FUNC(axpy)(const int *n, const RealScalar *palpha, const RealScalar *px, const int *incx, RealScalar *py, const int *incy)
{
  const Scalar* x = reinterpret_cast<const Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar alpha  = *reinterpret_cast<const Scalar*>(palpha);

  if(*n<=0) return 0;

  if(*incx==1 && *incy==1)    make_vector(y,*n) += alpha * make_vector(x,*n);
  else if(*incx>0 && *incy>0) make_vector(y,*n,*incy) += alpha * make_vector(x,*n,*incx);
  else if(*incx>0 && *incy<0) make_vector(y,*n,-*incy).reverse() += alpha * make_vector(x,*n,*incx);
  else if(*incx<0 && *incy>0) make_vector(y,*n,*incy) += alpha * make_vector(x,*n,-*incx).reverse();
  else if(*incx<0 && *incy<0) make_vector(y,*n,-*incy).reverse() += alpha * make_vector(x,*n,-*incx).reverse();

  return 0;
}

int EIGEN_BLAS_FUNC(copy)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  // be carefull, *incx==0 is allowed !!
  if(*incx==1 && *incy==1)
    make_vector(y,*n) = make_vector(x,*n);
  else
  {
    if(*incx<0) x = x - (*n-1)*(*incx);
    if(*incy<0) y = y - (*n-1)*(*incy);
    for(int i=0;i<*n;++i)
    {
      *y = *x;
      x += *incx;
      y += *incy;
    }
  }

  return 0;
}

int EIGEN_CAT(EIGEN_CAT(i,SCALAR_SUFFIX),amax_)(int *n, RealScalar *px, int *incx)
{
  if(*n<=0) return 0;
  Scalar* x = reinterpret_cast<Scalar*>(px);

  DenseIndex ret;
  if(*incx==1)  make_vector(x,*n).cwiseAbs().maxCoeff(&ret);
  else          make_vector(x,*n,std::abs(*incx)).cwiseAbs().maxCoeff(&ret);
  return int(ret)+1;
}

int EIGEN_CAT(EIGEN_CAT(i,SCALAR_SUFFIX),amin_)(int *n, RealScalar *px, int *incx)
{
  if(*n<=0) return 0;
  Scalar* x = reinterpret_cast<Scalar*>(px);

  DenseIndex ret;
  if(*incx==1)  make_vector(x,*n).cwiseAbs().minCoeff(&ret);
  else          make_vector(x,*n,std::abs(*incx)).cwiseAbs().minCoeff(&ret);
  return int(ret)+1;
}

int EIGEN_BLAS_FUNC(rotg)(RealScalar *pa, RealScalar *pb, RealScalar *pc, RealScalar *ps)
{
  using std::sqrt;
  using std::abs;

  Scalar& a = *reinterpret_cast<Scalar*>(pa);
  Scalar& b = *reinterpret_cast<Scalar*>(pb);
  RealScalar* c = pc;
  Scalar* s = reinterpret_cast<Scalar*>(ps);

  #if !ISCOMPLEX
  Scalar r,z;
  Scalar aa = abs(a);
  Scalar ab = abs(b);
  if((aa+ab)==Scalar(0))
  {
    *c = 1;
    *s = 0;
    r = 0;
    z = 0;
  }
  else
  {
    r = sqrt(a*a + b*b);
    Scalar amax = aa>ab ? a : b;
    r = amax>0 ? r : -r;
    *c = a/r;
    *s = b/r;
    z = 1;
    if (aa > ab) z = *s;
    if (ab > aa && *c!=RealScalar(0))
      z = Scalar(1)/ *c;
  }
  *pa = r;
  *pb = z;
  #else
  Scalar alpha;
  RealScalar norm,scale;
  if(abs(a)==RealScalar(0))
  {
    *c = RealScalar(0);
    *s = Scalar(1);
    a = b;
  }
  else
  {
    scale = abs(a) + abs(b);
    norm = scale*sqrt((numext::abs2(a/scale)) + (numext::abs2(b/scale)));
    alpha = a/abs(a);
    *c = abs(a)/norm;
    *s = alpha*numext::conj(b)/norm;
    a = alpha*norm;
  }
  #endif

//   JacobiRotation<Scalar> r;
//   r.makeGivens(a,b);
//   *c = r.c();
//   *s = r.s();

  return 0;
}

int EIGEN_BLAS_FUNC(scal)(int *n, RealScalar *palpha, RealScalar *px, int *incx)
{
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  if(*incx==1)  make_vector(x,*n) *= alpha;
  else          make_vector(x,*n,std::abs(*incx)) *= alpha;

  return 0;
}

int EIGEN_BLAS_FUNC(swap)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    make_vector(y,*n).swap(make_vector(x,*n));
  else if(*incx>0 && *incy>0) make_vector(y,*n,*incy).swap(make_vector(x,*n,*incx));
  else if(*incx>0 && *incy<0) make_vector(y,*n,-*incy).reverse().swap(make_vector(x,*n,*incx));
  else if(*incx<0 && *incy>0) make_vector(y,*n,*incy).swap(make_vector(x,*n,-*incx).reverse());
  else if(*incx<0 && *incy<0) make_vector(y,*n,-*incy).reverse().swap(make_vector(x,*n,-*incx).reverse());

  return 1;
}
