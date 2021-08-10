// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"
#include <Eigen/PaStiXSupport>
#include <unsupported/Eigen/SparseExtra>


template<typename T> void test_pastix_T()
{
  PastixLLT< SparseMatrix<T, ColMajor>, Eigen::Lower > pastix_llt_lower;
  PastixLDLT< SparseMatrix<T, ColMajor>, Eigen::Lower > pastix_ldlt_lower;
  PastixLLT< SparseMatrix<T, ColMajor>, Eigen::Upper > pastix_llt_upper;
  PastixLDLT< SparseMatrix<T, ColMajor>, Eigen::Upper > pastix_ldlt_upper;
  PastixLU< SparseMatrix<T, ColMajor> > pastix_lu;

  check_sparse_spd_solving(pastix_llt_lower);
  check_sparse_spd_solving(pastix_ldlt_lower);
  check_sparse_spd_solving(pastix_llt_upper);
  check_sparse_spd_solving(pastix_ldlt_upper);
  check_sparse_square_solving(pastix_lu);

  // Some compilation check:
  pastix_llt_lower.iparm();
  pastix_llt_lower.dparm();
  pastix_ldlt_lower.iparm();
  pastix_ldlt_lower.dparm();
  pastix_lu.iparm();
  pastix_lu.dparm();
}

// There is no support for selfadjoint matrices with PaStiX. 
// Complex symmetric matrices should pass though
template<typename T> void test_pastix_T_LU()
{
  PastixLU< SparseMatrix<T, ColMajor> > pastix_lu;
  check_sparse_square_solving(pastix_lu);
}

void test_pastix_support()
{
  CALL_SUBTEST_1(test_pastix_T<float>());
  CALL_SUBTEST_2(test_pastix_T<double>());
  CALL_SUBTEST_3( (test_pastix_T_LU<std::complex<float> >()) );
  CALL_SUBTEST_4(test_pastix_T_LU<std::complex<double> >());
} 
