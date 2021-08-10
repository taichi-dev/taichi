// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLAS_COMMON_H
#define EIGEN_BLAS_COMMON_H

#include "../Eigen/Core"
#include "../Eigen/Jacobi"

#include <complex>

#ifndef SCALAR
#error the token SCALAR must be defined to compile this file
#endif

#include "../Eigen/src/misc/blas.h"

#define NOTR    0
#define TR      1
#define ADJ     2

#define LEFT    0
#define RIGHT   1

#define UP      0
#define LO      1

#define NUNIT   0
#define UNIT    1

#define INVALID 0xff

#define OP(X)   (   ((X)=='N' || (X)=='n') ? NOTR   \
                  : ((X)=='T' || (X)=='t') ? TR     \
                  : ((X)=='C' || (X)=='c') ? ADJ    \
                  : INVALID)

#define SIDE(X) (   ((X)=='L' || (X)=='l') ? LEFT   \
                  : ((X)=='R' || (X)=='r') ? RIGHT  \
                  : INVALID)

#define UPLO(X) (   ((X)=='U' || (X)=='u') ? UP     \
                  : ((X)=='L' || (X)=='l') ? LO     \
                  : INVALID)

#define DIAG(X) (   ((X)=='N' || (X)=='n') ? NUNIT  \
                  : ((X)=='U' || (X)=='u') ? UNIT   \
                  : INVALID)


inline bool check_op(const char* op)
{
  return OP(*op)!=0xff;
}

inline bool check_side(const char* side)
{
  return SIDE(*side)!=0xff;
}

inline bool check_uplo(const char* uplo)
{
  return UPLO(*uplo)!=0xff;
}


namespace Eigen {
#include "BandTriangularSolver.h"
#include "GeneralRank1Update.h"
#include "PackedSelfadjointProduct.h"
#include "PackedTriangularMatrixVector.h"
#include "PackedTriangularSolverVector.h"
#include "Rank2Update.h"
}

using namespace Eigen;

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef std::complex<RealScalar> Complex;

enum
{
  IsComplex = Eigen::NumTraits<SCALAR>::IsComplex,
  Conj = IsComplex
};

typedef Matrix<Scalar,Dynamic,Dynamic,ColMajor> PlainMatrixType;
typedef Map<Matrix<Scalar,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> > MatrixType;
typedef Map<const Matrix<Scalar,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> > ConstMatrixType;
typedef Map<Matrix<Scalar,Dynamic,1>, 0, InnerStride<Dynamic> > StridedVectorType;
typedef Map<Matrix<Scalar,Dynamic,1> > CompactVectorType;

template<typename T>
Map<Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >
matrix(T* data, int rows, int cols, int stride)
{
  return Map<Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >(data, rows, cols, OuterStride<>(stride));
}

template<typename T>
Map<const Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >
matrix(const T* data, int rows, int cols, int stride)
{
  return Map<const Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >(data, rows, cols, OuterStride<>(stride));
}

template<typename T>
Map<Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> > make_vector(T* data, int size, int incr)
{
  return Map<Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> >(data, size, InnerStride<Dynamic>(incr));
}

template<typename T>
Map<const Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> > make_vector(const T* data, int size, int incr)
{
  return Map<const Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> >(data, size, InnerStride<Dynamic>(incr));
}

template<typename T>
Map<Matrix<T,Dynamic,1> > make_vector(T* data, int size)
{
  return Map<Matrix<T,Dynamic,1> >(data, size);
}

template<typename T>
Map<const Matrix<T,Dynamic,1> > make_vector(const T* data, int size)
{
  return Map<const Matrix<T,Dynamic,1> >(data, size);
}

template<typename T>
T* get_compact_vector(T* x, int n, int incx)
{
  if(incx==1)
    return x;

  typename Eigen::internal::remove_const<T>::type* ret = new Scalar[n];
  if(incx<0) make_vector(ret,n) = make_vector(x,n,-incx).reverse();
  else       make_vector(ret,n) = make_vector(x,n, incx);
  return ret;
}

template<typename T>
T* copy_back(T* x_cpy, T* x, int n, int incx)
{
  if(x_cpy==x)
    return 0;

  if(incx<0) make_vector(x,n,-incx).reverse() = make_vector(x_cpy,n);
  else       make_vector(x,n, incx)           = make_vector(x_cpy,n);
  return x_cpy;
}

#define EIGEN_BLAS_FUNC(X) EIGEN_CAT(SCALAR_SUFFIX,X##_)

#endif // EIGEN_BLAS_COMMON_H
