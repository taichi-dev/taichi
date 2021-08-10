// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define SCALAR        std::complex<float>
#define SCALAR_SUFFIX c
#define SCALAR_SUFFIX_UP "C"
#define REAL_SCALAR_SUFFIX s
#define ISCOMPLEX     1

#include "cholesky.cpp"
#include "lu.cpp"
#include "svd.cpp"
