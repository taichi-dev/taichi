// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse_solver.h"
#include <Eigen/SparseLU>
#include <Eigen/MetisSupport>
#include <unsupported/Eigen/SparseExtra>

template<typename T> void test_metis_T()
{
  SparseLU<SparseMatrix<T, ColMajor>, MetisOrdering<int> > sparselu_metis;
  
  check_sparse_square_solving(sparselu_metis); 
}

void test_metis_support()
{
  CALL_SUBTEST_1(test_metis_T<double>());
}
