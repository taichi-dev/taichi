// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKED_TRIANGULAR_MATRIX_VECTOR_H
#define EIGEN_PACKED_TRIANGULAR_MATRIX_VECTOR_H

namespace internal {

template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, bool ConjRhs, int StorageOrder>
struct packed_triangular_matrix_vector_product;

template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, bool ConjRhs>
struct packed_triangular_matrix_vector_product<Index,Mode,LhsScalar,ConjLhs,RhsScalar,ConjRhs,ColMajor>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  enum {
    IsLower     = (Mode & Lower)   ==Lower,
    HasUnitDiag = (Mode & UnitDiag)==UnitDiag,
    HasZeroDiag = (Mode & ZeroDiag)==ZeroDiag
  };
  static void run(Index size, const LhsScalar* lhs, const RhsScalar* rhs, ResScalar* res, ResScalar alpha)
  {
    internal::conj_if<ConjRhs> cj;
    typedef Map<const Matrix<LhsScalar,Dynamic,1> > LhsMap;
    typedef typename conj_expr_if<ConjLhs,LhsMap>::type ConjLhsType;
    typedef Map<Matrix<ResScalar,Dynamic,1> > ResMap;

    for (Index i=0; i<size; ++i)
    {
      Index s = IsLower&&(HasUnitDiag||HasZeroDiag) ? 1 : 0;
      Index r = IsLower ? size-i: i+1;
      if (EIGEN_IMPLIES(HasUnitDiag||HasZeroDiag, (--r)>0))
	ResMap(res+(IsLower ? s+i : 0),r) += alpha * cj(rhs[i]) * ConjLhsType(LhsMap(lhs+s,r));
      if (HasUnitDiag)
	res[i] += alpha * cj(rhs[i]);
      lhs += IsLower ? size-i: i+1;
    }
  };
};

template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, bool ConjRhs>
struct packed_triangular_matrix_vector_product<Index,Mode,LhsScalar,ConjLhs,RhsScalar,ConjRhs,RowMajor>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  enum {
    IsLower     = (Mode & Lower)   ==Lower,
    HasUnitDiag = (Mode & UnitDiag)==UnitDiag,
    HasZeroDiag = (Mode & ZeroDiag)==ZeroDiag
  };
  static void run(Index size, const LhsScalar* lhs, const RhsScalar* rhs, ResScalar* res, ResScalar alpha)
  {
    internal::conj_if<ConjRhs> cj;
    typedef Map<const Matrix<LhsScalar,Dynamic,1> > LhsMap;
    typedef typename conj_expr_if<ConjLhs,LhsMap>::type ConjLhsType;
    typedef Map<const Matrix<RhsScalar,Dynamic,1> > RhsMap;
    typedef typename conj_expr_if<ConjRhs,RhsMap>::type ConjRhsType;

    for (Index i=0; i<size; ++i)
    {
      Index s = !IsLower&&(HasUnitDiag||HasZeroDiag) ? 1 : 0;
      Index r = IsLower ? i+1 : size-i;
      if (EIGEN_IMPLIES(HasUnitDiag||HasZeroDiag, (--r)>0))
	res[i] += alpha * (ConjLhsType(LhsMap(lhs+s,r)).cwiseProduct(ConjRhsType(RhsMap(rhs+(IsLower ? 0 : s+i),r)))).sum();
      if (HasUnitDiag)
	res[i] += alpha * cj(rhs[i]);
      lhs += IsLower ? i+1 : size-i;
    }
  };
};

} // end namespace internal

#endif // EIGEN_PACKED_TRIANGULAR_MATRIX_VECTOR_H
