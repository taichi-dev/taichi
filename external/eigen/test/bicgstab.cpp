// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template<typename T, typename I> void test_bicgstab_T()
{
  BiCGSTAB<SparseMatrix<T,0,I>, DiagonalPreconditioner<T> >     bicgstab_colmajor_diag;
  BiCGSTAB<SparseMatrix<T,0,I>, IdentityPreconditioner    >     bicgstab_colmajor_I;
  BiCGSTAB<SparseMatrix<T,0,I>, IncompleteLUT<T,I> >              bicgstab_colmajor_ilut;
  //BiCGSTAB<SparseMatrix<T>, SSORPreconditioner<T> >     bicgstab_colmajor_ssor;

  bicgstab_colmajor_diag.setTolerance(NumTraits<T>::epsilon()*4);
  bicgstab_colmajor_ilut.setTolerance(NumTraits<T>::epsilon()*4);
  
  CALL_SUBTEST( check_sparse_square_solving(bicgstab_colmajor_diag)  );
//   CALL_SUBTEST( check_sparse_square_solving(bicgstab_colmajor_I)     );
  CALL_SUBTEST( check_sparse_square_solving(bicgstab_colmajor_ilut)     );
  //CALL_SUBTEST( check_sparse_square_solving(bicgstab_colmajor_ssor)     );
}

void test_bicgstab()
{
  CALL_SUBTEST_1((test_bicgstab_T<double,int>()) );
  CALL_SUBTEST_2((test_bicgstab_T<std::complex<double>, int>()));
  CALL_SUBTEST_3((test_bicgstab_T<double,long int>()));
}
