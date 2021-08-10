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

#include <Eigen/UmfPackSupport>

template<typename T> void test_umfpack_support_T()
{
  UmfPackLU<SparseMatrix<T, ColMajor> > umfpack_colmajor;
  UmfPackLU<SparseMatrix<T, RowMajor> > umfpack_rowmajor;
  
  check_sparse_square_solving(umfpack_colmajor);
  check_sparse_square_solving(umfpack_rowmajor);
  
  check_sparse_square_determinant(umfpack_colmajor);
  check_sparse_square_determinant(umfpack_rowmajor);
}

void test_umfpack_support()
{
  CALL_SUBTEST_1(test_umfpack_support_T<double>());
  CALL_SUBTEST_2(test_umfpack_support_T<std::complex<double> >());
}

