// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse_solver.h"

template<typename T, typename I> void test_simplicial_cholesky_T()
{
  typedef SparseMatrix<T,0,I> SparseMatrixType;
  SimplicialCholesky<SparseMatrixType, Lower> chol_colmajor_lower_amd;
  SimplicialCholesky<SparseMatrixType, Upper> chol_colmajor_upper_amd;
  SimplicialLLT<     SparseMatrixType, Lower> llt_colmajor_lower_amd;
  SimplicialLLT<     SparseMatrixType, Upper> llt_colmajor_upper_amd;
  SimplicialLDLT<    SparseMatrixType, Lower> ldlt_colmajor_lower_amd;
  SimplicialLDLT<    SparseMatrixType, Upper> ldlt_colmajor_upper_amd;
  SimplicialLDLT<    SparseMatrixType, Lower, NaturalOrdering<I> > ldlt_colmajor_lower_nat;
  SimplicialLDLT<    SparseMatrixType, Upper, NaturalOrdering<I> > ldlt_colmajor_upper_nat;

  check_sparse_spd_solving(chol_colmajor_lower_amd);
  check_sparse_spd_solving(chol_colmajor_upper_amd);
  check_sparse_spd_solving(llt_colmajor_lower_amd);
  check_sparse_spd_solving(llt_colmajor_upper_amd);
  check_sparse_spd_solving(ldlt_colmajor_lower_amd);
  check_sparse_spd_solving(ldlt_colmajor_upper_amd);
  
  check_sparse_spd_determinant(chol_colmajor_lower_amd);
  check_sparse_spd_determinant(chol_colmajor_upper_amd);
  check_sparse_spd_determinant(llt_colmajor_lower_amd);
  check_sparse_spd_determinant(llt_colmajor_upper_amd);
  check_sparse_spd_determinant(ldlt_colmajor_lower_amd);
  check_sparse_spd_determinant(ldlt_colmajor_upper_amd);
  
  check_sparse_spd_solving(ldlt_colmajor_lower_nat, 300, 1000);
  check_sparse_spd_solving(ldlt_colmajor_upper_nat, 300, 1000);
}

void test_simplicial_cholesky()
{
  CALL_SUBTEST_1(( test_simplicial_cholesky_T<double,int>() ));
  CALL_SUBTEST_2(( test_simplicial_cholesky_T<std::complex<double>, int>() ));
  CALL_SUBTEST_3(( test_simplicial_cholesky_T<double,long int>() ));
}
