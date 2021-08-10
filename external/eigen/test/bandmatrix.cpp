// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void bandmatrix(const MatrixType& _m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrixType;

  Index rows = _m.rows();
  Index cols = _m.cols();
  Index supers = _m.supers();
  Index subs = _m.subs();

  MatrixType m(rows,cols,supers,subs);

  DenseMatrixType dm1(rows,cols);
  dm1.setZero();

  m.diagonal().setConstant(123);
  dm1.diagonal().setConstant(123);
  for (int i=1; i<=m.supers();++i)
  {
    m.diagonal(i).setConstant(static_cast<RealScalar>(i));
    dm1.diagonal(i).setConstant(static_cast<RealScalar>(i));
  }
  for (int i=1; i<=m.subs();++i)
  {
    m.diagonal(-i).setConstant(-static_cast<RealScalar>(i));
    dm1.diagonal(-i).setConstant(-static_cast<RealScalar>(i));
  }
  //std::cerr << m.m_data << "\n\n" << m.toDense() << "\n\n" << dm1 << "\n\n\n\n";
  VERIFY_IS_APPROX(dm1,m.toDenseMatrix());

  for (int i=0; i<cols; ++i)
  {
    m.col(i).setConstant(static_cast<RealScalar>(i+1));
    dm1.col(i).setConstant(static_cast<RealScalar>(i+1));
  }
  Index d = (std::min)(rows,cols);
  Index a = std::max<Index>(0,cols-d-supers);
  Index b = std::max<Index>(0,rows-d-subs);
  if(a>0) dm1.block(0,d+supers,rows,a).setZero();
  dm1.block(0,supers+1,cols-supers-1-a,cols-supers-1-a).template triangularView<Upper>().setZero();
  dm1.block(subs+1,0,rows-subs-1-b,rows-subs-1-b).template triangularView<Lower>().setZero();
  if(b>0) dm1.block(d+subs,0,b,cols).setZero();
  //std::cerr << m.m_data << "\n\n" << m.toDense() << "\n\n" << dm1 << "\n\n";
  VERIFY_IS_APPROX(dm1,m.toDenseMatrix());

}

using Eigen::internal::BandMatrix;

void test_bandmatrix()
{
  for(int i = 0; i < 10*g_repeat ; i++) {
    Index rows = internal::random<Index>(1,10);
    Index cols = internal::random<Index>(1,10);
    Index sups = internal::random<Index>(0,cols-1);
    Index subs = internal::random<Index>(0,rows-1);
    CALL_SUBTEST(bandmatrix(BandMatrix<float>(rows,cols,sups,subs)) );
  }
}
