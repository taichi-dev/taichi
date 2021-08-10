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

#include <Eigen/CholmodSupport>

template<typename T> void test_cholmod_T()
{
  CholmodDecomposition<SparseMatrix<T>, Lower> g_chol_colmajor_lower; g_chol_colmajor_lower.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseMatrix<T>, Upper> g_chol_colmajor_upper; g_chol_colmajor_upper.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseMatrix<T>, Lower> g_llt_colmajor_lower;  g_llt_colmajor_lower.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseMatrix<T>, Upper> g_llt_colmajor_upper;  g_llt_colmajor_upper.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseMatrix<T>, Lower> g_ldlt_colmajor_lower; g_ldlt_colmajor_lower.setMode(CholmodLDLt);
  CholmodDecomposition<SparseMatrix<T>, Upper> g_ldlt_colmajor_upper; g_ldlt_colmajor_upper.setMode(CholmodLDLt);
  
  CholmodSupernodalLLT<SparseMatrix<T>, Lower> chol_colmajor_lower;
  CholmodSupernodalLLT<SparseMatrix<T>, Upper> chol_colmajor_upper;
  CholmodSimplicialLLT<SparseMatrix<T>, Lower> llt_colmajor_lower;
  CholmodSimplicialLLT<SparseMatrix<T>, Upper> llt_colmajor_upper;
  CholmodSimplicialLDLT<SparseMatrix<T>, Lower> ldlt_colmajor_lower;
  CholmodSimplicialLDLT<SparseMatrix<T>, Upper> ldlt_colmajor_upper;

  check_sparse_spd_solving(g_chol_colmajor_lower);
  check_sparse_spd_solving(g_chol_colmajor_upper);
  check_sparse_spd_solving(g_llt_colmajor_lower);
  check_sparse_spd_solving(g_llt_colmajor_upper);
  check_sparse_spd_solving(g_ldlt_colmajor_lower);
  check_sparse_spd_solving(g_ldlt_colmajor_upper);
  
  check_sparse_spd_solving(chol_colmajor_lower);
  check_sparse_spd_solving(chol_colmajor_upper);
  check_sparse_spd_solving(llt_colmajor_lower);
  check_sparse_spd_solving(llt_colmajor_upper);
  check_sparse_spd_solving(ldlt_colmajor_lower);
  check_sparse_spd_solving(ldlt_colmajor_upper);

  check_sparse_spd_determinant(chol_colmajor_lower);
  check_sparse_spd_determinant(chol_colmajor_upper);
  check_sparse_spd_determinant(llt_colmajor_lower);
  check_sparse_spd_determinant(llt_colmajor_upper);
  check_sparse_spd_determinant(ldlt_colmajor_lower);
  check_sparse_spd_determinant(ldlt_colmajor_upper);
}

void test_cholmod_support()
{
  CALL_SUBTEST_1(test_cholmod_T<double>());
  CALL_SUBTEST_2(test_cholmod_T<std::complex<double> >());
}
