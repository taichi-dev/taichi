// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// SparseLU solve does not accept column major matrices for the destination.
// However, as expected, the generic check_sparse_square_solving routines produces row-major
// rhs and destination matrices when compiled with EIGEN_DEFAULT_TO_ROW_MAJOR

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#undef EIGEN_DEFAULT_TO_ROW_MAJOR
#endif

#include "sparse_solver.h"
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>

template<typename T> void test_sparselu_T()
{
  SparseLU<SparseMatrix<T, ColMajor> /*, COLAMDOrdering<int>*/ > sparselu_colamd; // COLAMDOrdering is the default
  SparseLU<SparseMatrix<T, ColMajor>, AMDOrdering<int> > sparselu_amd; 
  SparseLU<SparseMatrix<T, ColMajor, long int>, NaturalOrdering<long int> > sparselu_natural;
  
  check_sparse_square_solving(sparselu_colamd,  300, 100000, true); 
  check_sparse_square_solving(sparselu_amd,     300,  10000, true);
  check_sparse_square_solving(sparselu_natural, 300,   2000, true);
  
  check_sparse_square_abs_determinant(sparselu_colamd);
  check_sparse_square_abs_determinant(sparselu_amd);
  
  check_sparse_square_determinant(sparselu_colamd);
  check_sparse_square_determinant(sparselu_amd);
}

void test_sparselu()
{
  CALL_SUBTEST_1(test_sparselu_T<float>()); 
  CALL_SUBTEST_2(test_sparselu_T<double>());
  CALL_SUBTEST_3(test_sparselu_T<std::complex<float> >()); 
  CALL_SUBTEST_4(test_sparselu_T<std::complex<double> >());
}
