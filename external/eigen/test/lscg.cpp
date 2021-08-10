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

template<typename T> void test_lscg_T()
{
  LeastSquaresConjugateGradient<SparseMatrix<T> > lscg_colmajor_diag;
  LeastSquaresConjugateGradient<SparseMatrix<T>, IdentityPreconditioner> lscg_colmajor_I;
  LeastSquaresConjugateGradient<SparseMatrix<T,RowMajor> > lscg_rowmajor_diag;
  LeastSquaresConjugateGradient<SparseMatrix<T,RowMajor>, IdentityPreconditioner> lscg_rowmajor_I;

  CALL_SUBTEST( check_sparse_square_solving(lscg_colmajor_diag)  );
  CALL_SUBTEST( check_sparse_square_solving(lscg_colmajor_I)     );
  
  CALL_SUBTEST( check_sparse_leastsquare_solving(lscg_colmajor_diag)  );
  CALL_SUBTEST( check_sparse_leastsquare_solving(lscg_colmajor_I)     );

  CALL_SUBTEST( check_sparse_square_solving(lscg_rowmajor_diag)  );
  CALL_SUBTEST( check_sparse_square_solving(lscg_rowmajor_I)     );

  CALL_SUBTEST( check_sparse_leastsquare_solving(lscg_rowmajor_diag)  );
  CALL_SUBTEST( check_sparse_leastsquare_solving(lscg_rowmajor_I)     );
}

void test_lscg()
{
  CALL_SUBTEST_1(test_lscg_T<double>());
  CALL_SUBTEST_2(test_lscg_T<std::complex<double> >());
}
