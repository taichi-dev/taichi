/* 
   Intel Copyright (C) ....
*/

#include "sparse_solver.h"
#include <Eigen/PardisoSupport>

template<typename T> void test_pardiso_T()
{
  PardisoLLT < SparseMatrix<T, RowMajor>, Lower> pardiso_llt_lower;
  PardisoLLT < SparseMatrix<T, RowMajor>, Upper> pardiso_llt_upper;
  PardisoLDLT < SparseMatrix<T, RowMajor>, Lower> pardiso_ldlt_lower;
  PardisoLDLT < SparseMatrix<T, RowMajor>, Upper> pardiso_ldlt_upper;
  PardisoLU  < SparseMatrix<T, RowMajor> > pardiso_lu;

  check_sparse_spd_solving(pardiso_llt_lower);
  check_sparse_spd_solving(pardiso_llt_upper);
  check_sparse_spd_solving(pardiso_ldlt_lower);
  check_sparse_spd_solving(pardiso_ldlt_upper);
  check_sparse_square_solving(pardiso_lu);
}

void test_pardiso_support()
{
  CALL_SUBTEST_1(test_pardiso_T<float>());
  CALL_SUBTEST_2(test_pardiso_T<double>());
  CALL_SUBTEST_3(test_pardiso_T< std::complex<float> >());
  CALL_SUBTEST_4(test_pardiso_T< std::complex<double> >());
}
