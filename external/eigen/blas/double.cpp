// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define SCALAR        double
#define SCALAR_SUFFIX d
#define SCALAR_SUFFIX_UP "D"
#define ISCOMPLEX     0

#include "level1_impl.h"
#include "level1_real_impl.h"
#include "level2_impl.h"
#include "level2_real_impl.h"
#include "level3_impl.h"

double BLASFUNC(dsdot)(int* n, float* x, int* incx, float* y, int* incy)
{
  if(*n<=0) return 0;

  if(*incx==1 && *incy==1)    return (make_vector(x,*n).cast<double>().cwiseProduct(make_vector(y,*n).cast<double>())).sum();
  else if(*incx>0 && *incy>0) return (make_vector(x,*n,*incx).cast<double>().cwiseProduct(make_vector(y,*n,*incy).cast<double>())).sum();
  else if(*incx<0 && *incy>0) return (make_vector(x,*n,-*incx).reverse().cast<double>().cwiseProduct(make_vector(y,*n,*incy).cast<double>())).sum();
  else if(*incx>0 && *incy<0) return (make_vector(x,*n,*incx).cast<double>().cwiseProduct(make_vector(y,*n,-*incy).reverse().cast<double>())).sum();
  else if(*incx<0 && *incy<0) return (make_vector(x,*n,-*incx).reverse().cast<double>().cwiseProduct(make_vector(y,*n,-*incy).reverse().cast<double>())).sum();
  else return 0;
}
