// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"

#include <Eigen/SuperLUSupport>

void test_superlu_support()
{
  SuperLU<SparseMatrix<double> > superlu_double_colmajor;
  SuperLU<SparseMatrix<std::complex<double> > > superlu_cplxdouble_colmajor;
  CALL_SUBTEST_1( check_sparse_square_solving(superlu_double_colmajor)      );
  CALL_SUBTEST_2( check_sparse_square_solving(superlu_cplxdouble_colmajor)  );
  CALL_SUBTEST_1( check_sparse_square_determinant(superlu_double_colmajor)      );
  CALL_SUBTEST_2( check_sparse_square_determinant(superlu_cplxdouble_colmajor)  );
}
