// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "common.h"

struct scalar_norm1_op {
  typedef RealScalar result_type;
  EIGEN_EMPTY_STRUCT_CTOR(scalar_norm1_op)
  inline RealScalar operator() (const Scalar& a) const { return numext::norm1(a); }
};
namespace Eigen {
  namespace internal {
    template<> struct functor_traits<scalar_norm1_op >
    {
      enum { Cost = 3 * NumTraits<Scalar>::AddCost, PacketAccess = 0 };
    };
  }
}

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
RealScalar EIGEN_CAT(EIGEN_CAT(REAL_SCALAR_SUFFIX,SCALAR_SUFFIX),asum_)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "__asum " << *n << " " << *incx << "\n";
  Complex* x = reinterpret_cast<Complex*>(px);

  if(*n<=0) return 0;

  if(*incx==1)  return make_vector(x,*n).unaryExpr<scalar_norm1_op>().sum();
  else          return make_vector(x,*n,std::abs(*incx)).unaryExpr<scalar_norm1_op>().sum();
}

// computes a dot product of a conjugated vector with another vector.
int EIGEN_BLAS_FUNC(dotcw)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar* pres)
{
//   std::cerr << "_dotc " << *n << " " << *incx << " " << *incy << "\n";
  Scalar* res = reinterpret_cast<Scalar*>(pres);

  if(*n<=0)
  {
    *res = Scalar(0);
    return 0;
  }

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    *res = (make_vector(x,*n).dot(make_vector(y,*n)));
  else if(*incx>0 && *incy>0) *res = (make_vector(x,*n,*incx).dot(make_vector(y,*n,*incy)));
  else if(*incx<0 && *incy>0) *res = (make_vector(x,*n,-*incx).reverse().dot(make_vector(y,*n,*incy)));
  else if(*incx>0 && *incy<0) *res = (make_vector(x,*n,*incx).dot(make_vector(y,*n,-*incy).reverse()));
  else if(*incx<0 && *incy<0) *res = (make_vector(x,*n,-*incx).reverse().dot(make_vector(y,*n,-*incy).reverse()));
  return 0;
}

// computes a vector-vector dot product without complex conjugation.
int EIGEN_BLAS_FUNC(dotuw)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar* pres)
{
  Scalar* res = reinterpret_cast<Scalar*>(pres);

  if(*n<=0)
  {
    *res = Scalar(0);
    return 0;
  }

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    *res = (make_vector(x,*n).cwiseProduct(make_vector(y,*n))).sum();
  else if(*incx>0 && *incy>0) *res = (make_vector(x,*n,*incx).cwiseProduct(make_vector(y,*n,*incy))).sum();
  else if(*incx<0 && *incy>0) *res = (make_vector(x,*n,-*incx).reverse().cwiseProduct(make_vector(y,*n,*incy))).sum();
  else if(*incx>0 && *incy<0) *res = (make_vector(x,*n,*incx).cwiseProduct(make_vector(y,*n,-*incy).reverse())).sum();
  else if(*incx<0 && *incy<0) *res = (make_vector(x,*n,-*incx).reverse().cwiseProduct(make_vector(y,*n,-*incy).reverse())).sum();
  return 0;
}

RealScalar EIGEN_CAT(EIGEN_CAT(REAL_SCALAR_SUFFIX,SCALAR_SUFFIX),nrm2_)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "__nrm2 " << *n << " " << *incx << "\n";
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*incx==1)
    return make_vector(x,*n).stableNorm();

  return make_vector(x,*n,*incx).stableNorm();
}

int EIGEN_CAT(EIGEN_CAT(SCALAR_SUFFIX,REAL_SCALAR_SUFFIX),rot_)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pc, RealScalar *ps)
{
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  RealScalar c = *pc;
  RealScalar s = *ps;

  StridedVectorType vx(make_vector(x,*n,std::abs(*incx)));
  StridedVectorType vy(make_vector(y,*n,std::abs(*incy)));

  Reverse<StridedVectorType> rvx(vx);
  Reverse<StridedVectorType> rvy(vy);

  // TODO implement mixed real-scalar rotations
       if(*incx<0 && *incy>0) internal::apply_rotation_in_the_plane(rvx, vy, JacobiRotation<Scalar>(c,s));
  else if(*incx>0 && *incy<0) internal::apply_rotation_in_the_plane(vx, rvy, JacobiRotation<Scalar>(c,s));
  else                        internal::apply_rotation_in_the_plane(vx, vy,  JacobiRotation<Scalar>(c,s));

  return 0;
}

int EIGEN_CAT(EIGEN_CAT(SCALAR_SUFFIX,REAL_SCALAR_SUFFIX),scal_)(int *n, RealScalar *palpha, RealScalar *px, int *incx)
{
  if(*n<=0) return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  RealScalar alpha = *palpha;

//   std::cerr << "__scal " << *n << " " << alpha << " " << *incx << "\n";

  if(*incx==1)  make_vector(x,*n) *= alpha;
  else          make_vector(x,*n,std::abs(*incx)) *= alpha;

  return 0;
}
