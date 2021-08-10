// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BAND_TRIANGULARSOLVER_H
#define EIGEN_BAND_TRIANGULARSOLVER_H

namespace internal {

 /* \internal
  * Solve Ax=b with A a band triangular matrix
  * TODO: extend it to matrices for x abd b */
template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, int StorageOrder>
struct band_solve_triangular_selector;


template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar>
struct band_solve_triangular_selector<Index,Mode,LhsScalar,ConjLhs,RhsScalar,RowMajor>
{
  typedef Map<const Matrix<LhsScalar,Dynamic,Dynamic,RowMajor>, 0, OuterStride<> > LhsMap;
  typedef Map<Matrix<RhsScalar,Dynamic,1> > RhsMap;
  enum { IsLower = (Mode&Lower) ? 1 : 0 };
  static void run(Index size, Index k, const LhsScalar* _lhs, Index lhsStride, RhsScalar* _other)
  {
    const LhsMap lhs(_lhs,size,k+1,OuterStride<>(lhsStride));
    RhsMap other(_other,size,1);
    typename internal::conditional<
                          ConjLhs,
                          const CwiseUnaryOp<typename internal::scalar_conjugate_op<LhsScalar>,LhsMap>,
                          const LhsMap&>
                        ::type cjLhs(lhs);
                        
    for(int col=0 ; col<other.cols() ; ++col)
    {
      for(int ii=0; ii<size; ++ii)
      {
        int i = IsLower ? ii : size-ii-1;
        int actual_k = (std::min)(k,ii);
        int actual_start = IsLower ? k-actual_k : 1;
        
        if(actual_k>0)
          other.coeffRef(i,col) -= cjLhs.row(i).segment(actual_start,actual_k).transpose()
                                  .cwiseProduct(other.col(col).segment(IsLower ? i-actual_k : i+1,actual_k)).sum();

        if((Mode&UnitDiag)==0)
          other.coeffRef(i,col) /= cjLhs(i,IsLower ? k : 0);
      }
    }
  }
  
};

template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar>
struct band_solve_triangular_selector<Index,Mode,LhsScalar,ConjLhs,RhsScalar,ColMajor>
{
  typedef Map<const Matrix<LhsScalar,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> > LhsMap;
  typedef Map<Matrix<RhsScalar,Dynamic,1> > RhsMap;
  enum { IsLower = (Mode&Lower) ? 1 : 0 };
  static void run(Index size, Index k, const LhsScalar* _lhs, Index lhsStride, RhsScalar* _other)
  {
    const LhsMap lhs(_lhs,k+1,size,OuterStride<>(lhsStride));
    RhsMap other(_other,size,1);
    typename internal::conditional<
                          ConjLhs,
                          const CwiseUnaryOp<typename internal::scalar_conjugate_op<LhsScalar>,LhsMap>,
                          const LhsMap&>
                        ::type cjLhs(lhs);
                        
    for(int col=0 ; col<other.cols() ; ++col)
    {
      for(int ii=0; ii<size; ++ii)
      {
        int i = IsLower ? ii : size-ii-1;
        int actual_k = (std::min)(k,size-ii-1);
        int actual_start = IsLower ? 1 : k-actual_k;
        
        if((Mode&UnitDiag)==0)
          other.coeffRef(i,col) /= cjLhs(IsLower ? 0 : k, i);

        if(actual_k>0)
          other.col(col).segment(IsLower ? i+1 : i-actual_k, actual_k)
              -= other.coeff(i,col) * cjLhs.col(i).segment(actual_start,actual_k);
        
      }
    }
  }
};


} // end namespace internal

#endif // EIGEN_BAND_TRIANGULARSOLVER_H
