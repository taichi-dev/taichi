// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
RealScalar EIGEN_BLAS_FUNC(asum)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "_asum " << *n << " " << *incx << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*n<=0) return 0;

  if(*incx==1)  return make_vector(x,*n).cwiseAbs().sum();
  else          return make_vector(x,*n,std::abs(*incx)).cwiseAbs().sum();
}

// computes a vector-vector dot product.
Scalar EIGEN_BLAS_FUNC(dot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
//   std::cerr << "_dot " << *n << " " << *incx << " " << *incy << "\n";

  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    return (make_vector(x,*n).cwiseProduct(make_vector(y,*n))).sum();
  else if(*incx>0 && *incy>0) return (make_vector(x,*n,*incx).cwiseProduct(make_vector(y,*n,*incy))).sum();
  else if(*incx<0 && *incy>0) return (make_vector(x,*n,-*incx).reverse().cwiseProduct(make_vector(y,*n,*incy))).sum();
  else if(*incx>0 && *incy<0) return (make_vector(x,*n,*incx).cwiseProduct(make_vector(y,*n,-*incy).reverse())).sum();
  else if(*incx<0 && *incy<0) return (make_vector(x,*n,-*incx).reverse().cwiseProduct(make_vector(y,*n,-*incy).reverse())).sum();
  else return 0;
}

// computes the Euclidean norm of a vector.
// FIXME
Scalar EIGEN_BLAS_FUNC(nrm2)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "_nrm2 " << *n << " " << *incx << "\n";
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*incx==1)  return make_vector(x,*n).stableNorm();
  else          return make_vector(x,*n,std::abs(*incx)).stableNorm();
}

int EIGEN_BLAS_FUNC(rot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pc, RealScalar *ps)
{
//   std::cerr << "_rot " << *n << " " << *incx << " " << *incy << "\n";
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar c = *reinterpret_cast<Scalar*>(pc);
  Scalar s = *reinterpret_cast<Scalar*>(ps);

  StridedVectorType vx(make_vector(x,*n,std::abs(*incx)));
  StridedVectorType vy(make_vector(y,*n,std::abs(*incy)));

  Reverse<StridedVectorType> rvx(vx);
  Reverse<StridedVectorType> rvy(vy);

       if(*incx<0 && *incy>0) internal::apply_rotation_in_the_plane(rvx, vy, JacobiRotation<Scalar>(c,s));
  else if(*incx>0 && *incy<0) internal::apply_rotation_in_the_plane(vx, rvy, JacobiRotation<Scalar>(c,s));
  else                        internal::apply_rotation_in_the_plane(vx, vy,  JacobiRotation<Scalar>(c,s));


  return 0;
}

/*
// performs rotation of points in the modified plane.
int EIGEN_BLAS_FUNC(rotm)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *param)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  // TODO

  return 0;
}

// computes the modified parameters for a Givens rotation.
int EIGEN_BLAS_FUNC(rotmg)(RealScalar *d1, RealScalar *d2, RealScalar *x1, RealScalar *x2, RealScalar *param)
{
  // TODO

  return 0;
}
*/
