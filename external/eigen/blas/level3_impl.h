// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <iostream>
#include "common.h"

int EIGEN_BLAS_FUNC(gemm)(const char *opa, const char *opb, const int *m, const int *n, const int *k, const RealScalar *palpha,
                          const RealScalar *pa, const int *lda, const RealScalar *pb, const int *ldb, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
//   std::cerr << "in gemm " << *opa << " " << *opb << " " << *m << " " << *n << " " << *k << " " << *lda << " " << *ldb << " " << *ldc << " " << *palpha << " " << *pbeta << "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex, Scalar, internal::level3_blocking<Scalar,Scalar>&, Eigen::internal::GemmParallelInfo<DenseIndex>*);
  static const functype func[12] = {
    // array index: NOTR  | (NOTR << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (NOTR << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,ColMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (NOTR << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (TR   << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,false,ColMajor,1>::run),
    // array index: TR    | (TR   << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (TR   << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (ADJ  << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,Conj, ColMajor,1>::run),
    // array index: TR    | (ADJ  << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,Conj, ColMajor,1>::run),
    // array index: ADJ   | (ADJ  << 2)
    (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,Conj, ColMajor,1>::run),
    0
  };

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* b = reinterpret_cast<const Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha  = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta   = *reinterpret_cast<const Scalar*>(pbeta);

  int info = 0;
  if(OP(*opa)==INVALID)                                               info = 1;
  else if(OP(*opb)==INVALID)                                          info = 2;
  else if(*m<0)                                                       info = 3;
  else if(*n<0)                                                       info = 4;
  else if(*k<0)                                                       info = 5;
  else if(*lda<std::max(1,(OP(*opa)==NOTR)?*m:*k))                    info = 8;
  else if(*ldb<std::max(1,(OP(*opb)==NOTR)?*k:*n))                    info = 10;
  else if(*ldc<std::max(1,*m))                                        info = 13;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"GEMM ",&info,6);

  if (*m == 0 || *n == 0)
    return 0;

  if(beta!=Scalar(1))
  {
    if(beta==Scalar(0)) matrix(c, *m, *n, *ldc).setZero();
    else                matrix(c, *m, *n, *ldc) *= beta;
  }

  if(*k == 0)
    return 0;

  internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*m,*n,*k,1,true);

  int code = OP(*opa) | (OP(*opb) << 2);
  func[code](*m, *n, *k, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking, 0);
  return 0;
}

int EIGEN_BLAS_FUNC(trsm)(const char *side, const char *uplo, const char *opa, const char *diag, const int *m, const int *n,
                          const RealScalar *palpha,  const RealScalar *pa, const int *lda, RealScalar *pb, const int *ldb)
{
//   std::cerr << "in trsm " << *side << " " << *uplo << " " << *opa << " " << *diag << " " << *m << "," << *n << " " << *palpha << " " << *lda << " " << *ldb<< "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex, internal::level3_blocking<Scalar,Scalar>&);
  static const functype func[32] = {
    // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          Conj, RowMajor,ColMajor,1>::run),\
    0,
    // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,Conj, RowMajor,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,false,ColMajor,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,false,RowMajor,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,Conj, RowMajor,ColMajor,1>::run),
    0
  };

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<const Scalar*>(palpha);

  int info = 0;
  if(SIDE(*side)==INVALID)                                            info = 1;
  else if(UPLO(*uplo)==INVALID)                                       info = 2;
  else if(OP(*opa)==INVALID)                                          info = 3;
  else if(DIAG(*diag)==INVALID)                                       info = 4;
  else if(*m<0)                                                       info = 5;
  else if(*n<0)                                                       info = 6;
  else if(*lda<std::max(1,(SIDE(*side)==LEFT)?*m:*n))                 info = 9;
  else if(*ldb<std::max(1,*m))                                        info = 11;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"TRSM ",&info,6);

  if(*m==0 || *n==0)
    return 0;

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);

  if(SIDE(*side)==LEFT)
  {
    internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic,4> blocking(*m,*n,*m,1,false);
    func[code](*m, *n, a, *lda, b, 1, *ldb, blocking);
  }
  else
  {
    internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic,4> blocking(*m,*n,*n,1,false);
    func[code](*n, *m, a, *lda, b, 1, *ldb, blocking);
  }

  if(alpha!=Scalar(1))
    matrix(b,*m,*n,*ldb) *= alpha;

  return 0;
}


// b = alpha*op(a)*b  for side = 'L'or'l'
// b = alpha*b*op(a)  for side = 'R'or'r'
int EIGEN_BLAS_FUNC(trmm)(const char *side, const char *uplo, const char *opa, const char *diag, const int *m, const int *n,
                          const RealScalar *palpha, const RealScalar *pa, const int *lda, RealScalar *pb, const int *ldb)
{
//   std::cerr << "in trmm " << *side << " " << *uplo << " " << *opa << " " << *diag << " " << *m << " " << *n << " " << *lda << " " << *ldb << " " << *palpha << "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex, const Scalar&, internal::level3_blocking<Scalar,Scalar>&);
  static const functype func[32] = {
    // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, RowMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, RowMajor,Conj, ColMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,RowMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,RowMajor,Conj, ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, RowMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, RowMajor,Conj, ColMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,RowMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,RowMajor,Conj, ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor,1>::run),
    0,
    // array index: NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor,1>::run),
    0,
    // array index: NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor,1>::run),
    // array index: TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor,1>::run),
    // array index: ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)
    (internal::product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor,1>::run),
    0
  };

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<const Scalar*>(palpha);

  int info = 0;
  if(SIDE(*side)==INVALID)                                            info = 1;
  else if(UPLO(*uplo)==INVALID)                                       info = 2;
  else if(OP(*opa)==INVALID)                                          info = 3;
  else if(DIAG(*diag)==INVALID)                                       info = 4;
  else if(*m<0)                                                       info = 5;
  else if(*n<0)                                                       info = 6;
  else if(*lda<std::max(1,(SIDE(*side)==LEFT)?*m:*n))                 info = 9;
  else if(*ldb<std::max(1,*m))                                        info = 11;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"TRMM ",&info,6);

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);

  if(*m==0 || *n==0)
    return 1;

  // FIXME find a way to avoid this copy
  Matrix<Scalar,Dynamic,Dynamic,ColMajor> tmp = matrix(b,*m,*n,*ldb);
  matrix(b,*m,*n,*ldb).setZero();

  if(SIDE(*side)==LEFT)
  {
    internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic,4> blocking(*m,*n,*m,1,false);
    func[code](*m, *n, *m, a, *lda, tmp.data(), tmp.outerStride(), b, 1, *ldb, alpha, blocking);
  }
  else
  {
    internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic,4> blocking(*m,*n,*n,1,false);
    func[code](*m, *n, *n, tmp.data(), tmp.outerStride(), a, *lda, b, 1, *ldb, alpha, blocking);
  }
  return 1;
}

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(symm)(const char *side, const char *uplo, const int *m, const int *n, const RealScalar *palpha,
                          const RealScalar *pa, const int *lda, const RealScalar *pb, const int *ldb, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
//   std::cerr << "in symm " << *side << " " << *uplo << " " << *m << "x" << *n << " lda:" << *lda << " ldb:" << *ldb << " ldc:" << *ldc << " alpha:" << *palpha << " beta:" << *pbeta << "\n";
  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* b = reinterpret_cast<const Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<const Scalar*>(pbeta);

  int info = 0;
  if(SIDE(*side)==INVALID)                                            info = 1;
  else if(UPLO(*uplo)==INVALID)                                       info = 2;
  else if(*m<0)                                                       info = 3;
  else if(*n<0)                                                       info = 4;
  else if(*lda<std::max(1,(SIDE(*side)==LEFT)?*m:*n))                 info = 7;
  else if(*ldb<std::max(1,*m))                                        info = 9;
  else if(*ldc<std::max(1,*m))                                        info = 12;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"SYMM ",&info,6);

  if(beta!=Scalar(1))
  {
    if(beta==Scalar(0)) matrix(c, *m, *n, *ldc).setZero();
    else                matrix(c, *m, *n, *ldc) *= beta;
  }

  if(*m==0 || *n==0)
  {
    return 1;
  }

  int size = (SIDE(*side)==LEFT) ? (*m) : (*n);
  #if ISCOMPLEX
  // FIXME add support for symmetric complex matrix
  Matrix<Scalar,Dynamic,Dynamic,ColMajor> matA(size,size);
  if(UPLO(*uplo)==UP)
  {
    matA.triangularView<Upper>() = matrix(a,size,size,*lda);
    matA.triangularView<Lower>() = matrix(a,size,size,*lda).transpose();
  }
  else if(UPLO(*uplo)==LO)
  {
    matA.triangularView<Lower>() = matrix(a,size,size,*lda);
    matA.triangularView<Upper>() = matrix(a,size,size,*lda).transpose();
  }
  if(SIDE(*side)==LEFT)
    matrix(c, *m, *n, *ldc) += alpha * matA * matrix(b, *m, *n, *ldb);
  else if(SIDE(*side)==RIGHT)
    matrix(c, *m, *n, *ldc) += alpha * matrix(b, *m, *n, *ldb) * matA;
  #else
  internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*m,*n,size,1,false);

  if(SIDE(*side)==LEFT)
    if(UPLO(*uplo)==UP)       internal::product_selfadjoint_matrix<Scalar, DenseIndex, RowMajor,true,false, ColMajor,false,false, ColMajor,1>::run(*m, *n, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking);
    else if(UPLO(*uplo)==LO)  internal::product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,true,false, ColMajor,false,false, ColMajor,1>::run(*m, *n, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking);
    else                      return 0;
  else if(SIDE(*side)==RIGHT)
    if(UPLO(*uplo)==UP)       internal::product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,false,false, RowMajor,true,false, ColMajor,1>::run(*m, *n, b, *ldb, a, *lda, c, 1, *ldc, alpha, blocking);
    else if(UPLO(*uplo)==LO)  internal::product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,false,false, ColMajor,true,false, ColMajor,1>::run(*m, *n, b, *ldb, a, *lda, c, 1, *ldc, alpha, blocking);
    else                      return 0;
  else
    return 0;
  #endif

  return 0;
}

// c = alpha*a*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*a + beta*c  for op = 'T'or't','C'or'c'
int EIGEN_BLAS_FUNC(syrk)(const char *uplo, const char *op, const int *n, const int *k,
                          const RealScalar *palpha, const RealScalar *pa, const int *lda, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
//   std::cerr << "in syrk " << *uplo << " " << *op << " " << *n << " " << *k << " " << *palpha << " " << *lda << " " << *pbeta << " " << *ldc << "\n";
  #if !ISCOMPLEX
  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex, const Scalar&, internal::level3_blocking<Scalar,Scalar>&);
  static const functype func[8] = {
    // array index: NOTR  | (UP << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,ColMajor,Conj, 1, Upper>::run),
    // array index: TR    | (UP << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,false,Scalar,ColMajor,ColMajor,Conj, 1, Upper>::run),
    // array index: ADJ   | (UP << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,ColMajor,false,1, Upper>::run),
    0,
    // array index: NOTR  | (LO << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,ColMajor,Conj, 1, Lower>::run),
    // array index: TR    | (LO << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,false,Scalar,ColMajor,ColMajor,Conj, 1, Lower>::run),
    // array index: ADJ   | (LO << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,ColMajor,false,1, Lower>::run),
    0
  };
  #endif

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<const Scalar*>(pbeta);

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(OP(*op)==INVALID || (ISCOMPLEX && OP(*op)==ADJ) )           info = 2;
  else if(*n<0)                                                       info = 3;
  else if(*k<0)                                                       info = 4;
  else if(*lda<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 7;
  else if(*ldc<std::max(1,*n))                                        info = 10;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"SYRK ",&info,6);

  if(beta!=Scalar(1))
  {
    if(UPLO(*uplo)==UP)
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Upper>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<Upper>() *= beta;
    else
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Lower>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<Lower>() *= beta;
  }

  if(*n==0 || *k==0)
    return 0;

  #if ISCOMPLEX
  // FIXME add support for symmetric complex matrix
  if(UPLO(*uplo)==UP)
  {
    if(OP(*op)==NOTR)
      matrix(c, *n, *n, *ldc).triangularView<Upper>() += alpha * matrix(a,*n,*k,*lda) * matrix(a,*n,*k,*lda).transpose();
    else
      matrix(c, *n, *n, *ldc).triangularView<Upper>() += alpha * matrix(a,*k,*n,*lda).transpose() * matrix(a,*k,*n,*lda);
  }
  else
  {
    if(OP(*op)==NOTR)
      matrix(c, *n, *n, *ldc).triangularView<Lower>() += alpha * matrix(a,*n,*k,*lda) * matrix(a,*n,*k,*lda).transpose();
    else
      matrix(c, *n, *n, *ldc).triangularView<Lower>() += alpha * matrix(a,*k,*n,*lda).transpose() * matrix(a,*k,*n,*lda);
  }
  #else
  internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*n,*n,*k,1,false);

  int code = OP(*op) | (UPLO(*uplo) << 2);
  func[code](*n, *k, a, *lda, a, *lda, c, 1, *ldc, alpha, blocking);
  #endif

  return 0;
}

// c = alpha*a*b' + alpha*b*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*b + alpha*b'*a + beta*c  for op = 'T'or't'
int EIGEN_BLAS_FUNC(syr2k)(const char *uplo, const char *op, const int *n, const int *k, const RealScalar *palpha,
                           const RealScalar *pa, const int *lda, const RealScalar *pb, const int *ldb, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* b = reinterpret_cast<const Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<const Scalar*>(pbeta);

//   std::cerr << "in syr2k " << *uplo << " " << *op << " " << *n << " " << *k << " " << alpha << " " << *lda << " " << *ldb << " " << beta << " " << *ldc << "\n";

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if(OP(*op)==INVALID || (ISCOMPLEX && OP(*op)==ADJ) )           info = 2;
  else if(*n<0)                                                       info = 3;
  else if(*k<0)                                                       info = 4;
  else if(*lda<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 7;
  else if(*ldb<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 9;
  else if(*ldc<std::max(1,*n))                                        info = 12;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"SYR2K",&info,6);

  if(beta!=Scalar(1))
  {
    if(UPLO(*uplo)==UP)
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Upper>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<Upper>() *= beta;
    else
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Lower>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<Lower>() *= beta;
  }

  if(*k==0)
    return 1;

  if(OP(*op)==NOTR)
  {
    if(UPLO(*uplo)==UP)
    {
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        += alpha *matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).transpose()
        +  alpha*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).transpose();
    }
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).transpose()
        +  alpha*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).transpose();
  }
  else if(OP(*op)==TR || OP(*op)==ADJ)
  {
    if(UPLO(*uplo)==UP)
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        += alpha*matrix(a, *k, *n, *lda).transpose()*matrix(b, *k, *n, *ldb)
        +  alpha*matrix(b, *k, *n, *ldb).transpose()*matrix(a, *k, *n, *lda);
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *k, *n, *lda).transpose()*matrix(b, *k, *n, *ldb)
        +  alpha*matrix(b, *k, *n, *ldb).transpose()*matrix(a, *k, *n, *lda);
  }

  return 0;
}


#if ISCOMPLEX

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(hemm)(const char *side, const char *uplo, const int *m, const int *n, const RealScalar *palpha,
                          const RealScalar *pa, const int *lda, const RealScalar *pb, const int *ldb, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* b = reinterpret_cast<const Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<const Scalar*>(pbeta);

//   std::cerr << "in hemm " << *side << " " << *uplo << " " << *m << " " << *n << " " << alpha << " " << *lda << " " << beta << " " << *ldc << "\n";

  int info = 0;
  if(SIDE(*side)==INVALID)                                            info = 1;
  else if(UPLO(*uplo)==INVALID)                                       info = 2;
  else if(*m<0)                                                       info = 3;
  else if(*n<0)                                                       info = 4;
  else if(*lda<std::max(1,(SIDE(*side)==LEFT)?*m:*n))                 info = 7;
  else if(*ldb<std::max(1,*m))                                        info = 9;
  else if(*ldc<std::max(1,*m))                                        info = 12;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HEMM ",&info,6);

  if(beta==Scalar(0))       matrix(c, *m, *n, *ldc).setZero();
  else if(beta!=Scalar(1))  matrix(c, *m, *n, *ldc) *= beta;

  if(*m==0 || *n==0)
  {
    return 1;
  }

  int size = (SIDE(*side)==LEFT) ? (*m) : (*n);
  internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*m,*n,size,1,false);

  if(SIDE(*side)==LEFT)
  {
    if(UPLO(*uplo)==UP)       internal::product_selfadjoint_matrix<Scalar,DenseIndex,RowMajor,true,Conj,  ColMajor,false,false, ColMajor, 1>
                                ::run(*m, *n, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking);
    else if(UPLO(*uplo)==LO)  internal::product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,true,false, ColMajor,false,false, ColMajor,1>
                                ::run(*m, *n, a, *lda, b, *ldb, c, 1, *ldc, alpha, blocking);
    else                      return 0;
  }
  else if(SIDE(*side)==RIGHT)
  {
    if(UPLO(*uplo)==UP)       matrix(c,*m,*n,*ldc) += alpha * matrix(b,*m,*n,*ldb) * matrix(a,*n,*n,*lda).selfadjointView<Upper>();/*internal::product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,false,false, RowMajor,true,Conj,  ColMajor, 1>
                                ::run(*m, *n, b, *ldb, a, *lda, c, 1, *ldc, alpha, blocking);*/
    else if(UPLO(*uplo)==LO)  internal::product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,false,false, ColMajor,true,false, ColMajor,1>
                                ::run(*m, *n, b, *ldb, a, *lda, c, 1, *ldc, alpha, blocking);
    else                      return 0;
  }
  else
  {
    return 0;
  }

  return 0;
}

// c = alpha*a*conj(a') + beta*c  for op = 'N'or'n'
// c = alpha*conj(a')*a + beta*c  for op  = 'C'or'c'
int EIGEN_BLAS_FUNC(herk)(const char *uplo, const char *op, const int *n, const int *k,
                          const RealScalar *palpha, const RealScalar *pa, const int *lda, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
//   std::cerr << "in herk " << *uplo << " " << *op << " " << *n << " " << *k << " " << *palpha << " " << *lda << " " << *pbeta << " " << *ldc << "\n";

  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, DenseIndex, const Scalar&, internal::level3_blocking<Scalar,Scalar>&);
  static const functype func[8] = {
    // array index: NOTR  | (UP << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,Conj, ColMajor,1,Upper>::run),
    0,
    // array index: ADJ   | (UP << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,false,ColMajor,1,Upper>::run),
    0,
    // array index: NOTR  | (LO << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,Conj, ColMajor,1,Lower>::run),
    0,
    // array index: ADJ   | (LO << 2)
    (internal::general_matrix_matrix_triangular_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,false,ColMajor,1,Lower>::run),
    0
  };

  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  RealScalar alpha = *palpha;
  RealScalar beta  = *pbeta;

//   std::cerr << "in herk " << *uplo << " " << *op << " " << *n << " " << *k << " " << alpha << " " << *lda << " " << beta << " " << *ldc << "\n";

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if((OP(*op)==INVALID) || (OP(*op)==TR))                        info = 2;
  else if(*n<0)                                                       info = 3;
  else if(*k<0)                                                       info = 4;
  else if(*lda<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 7;
  else if(*ldc<std::max(1,*n))                                        info = 10;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HERK ",&info,6);

  int code = OP(*op) | (UPLO(*uplo) << 2);

  if(beta!=RealScalar(1))
  {
    if(UPLO(*uplo)==UP)
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Upper>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<StrictlyUpper>() *= beta;
    else
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Lower>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<StrictlyLower>() *= beta;

    if(beta!=Scalar(0))
    {
      matrix(c, *n, *n, *ldc).diagonal().real() *= beta;
      matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
    }
  }

  if(*k>0 && alpha!=RealScalar(0))
  {
    internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*n,*n,*k,1,false);
    func[code](*n, *k, a, *lda, a, *lda, c, 1, *ldc, alpha, blocking);
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  }
  return 0;
}

// c = alpha*a*conj(b') + conj(alpha)*b*conj(a') + beta*c,  for op = 'N'or'n'
// c = alpha*conj(a')*b + conj(alpha)*conj(b')*a + beta*c,  for op = 'C'or'c'
int EIGEN_BLAS_FUNC(her2k)(const char *uplo, const char *op, const int *n, const int *k,
                           const RealScalar *palpha, const RealScalar *pa, const int *lda, const RealScalar *pb, const int *ldb, const RealScalar *pbeta, RealScalar *pc, const int *ldc)
{
  const Scalar* a = reinterpret_cast<const Scalar*>(pa);
  const Scalar* b = reinterpret_cast<const Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<const Scalar*>(palpha);
  RealScalar beta  = *pbeta;

//   std::cerr << "in her2k " << *uplo << " " << *op << " " << *n << " " << *k << " " << alpha << " " << *lda << " " << *ldb << " " << beta << " " << *ldc << "\n";

  int info = 0;
  if(UPLO(*uplo)==INVALID)                                            info = 1;
  else if((OP(*op)==INVALID) || (OP(*op)==TR))                        info = 2;
  else if(*n<0)                                                       info = 3;
  else if(*k<0)                                                       info = 4;
  else if(*lda<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 7;
  else if(*ldb<std::max(1,(OP(*op)==NOTR)?*n:*k))                     info = 9;
  else if(*ldc<std::max(1,*n))                                        info = 12;
  if(info)
    return xerbla_(SCALAR_SUFFIX_UP"HER2K",&info,6);

  if(beta!=RealScalar(1))
  {
    if(UPLO(*uplo)==UP)
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Upper>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<StrictlyUpper>() *= beta;
    else
      if(beta==Scalar(0)) matrix(c, *n, *n, *ldc).triangularView<Lower>().setZero();
      else                matrix(c, *n, *n, *ldc).triangularView<StrictlyLower>() *= beta;

    if(beta!=Scalar(0))
    {
      matrix(c, *n, *n, *ldc).diagonal().real() *= beta;
      matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
    }
  }
  else if(*k>0 && alpha!=Scalar(0))
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();

  if(*k==0)
    return 1;

  if(OP(*op)==NOTR)
  {
    if(UPLO(*uplo)==UP)
    {
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        +=            alpha *matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).adjoint()
        +  numext::conj(alpha)*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).adjoint();
    }
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).adjoint()
        +  numext::conj(alpha)*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).adjoint();
  }
  else if(OP(*op)==ADJ)
  {
    if(UPLO(*uplo)==UP)
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        +=             alpha*matrix(a, *k, *n, *lda).adjoint()*matrix(b, *k, *n, *ldb)
        +  numext::conj(alpha)*matrix(b, *k, *n, *ldb).adjoint()*matrix(a, *k, *n, *lda);
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        +=             alpha*matrix(a, *k, *n, *lda).adjoint()*matrix(b, *k, *n, *ldb)
        +  numext::conj(alpha)*matrix(b, *k, *n, *ldb).adjoint()*matrix(a, *k, *n, *lda);
  }

  return 1;
}

#endif // ISCOMPLEX
