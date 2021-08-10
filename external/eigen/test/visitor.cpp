// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void matrixVisitor(const MatrixType& p)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = p.rows();
  Index cols = p.cols();

  // construct a random matrix where all coefficients are different
  MatrixType m;
  m = MatrixType::Random(rows, cols);
  for(Index i = 0; i < m.size(); i++)
    for(Index i2 = 0; i2 < i; i2++)
      while(m(i) == m(i2)) // yes, ==
        m(i) = internal::random<Scalar>();
  
  Scalar minc = Scalar(1000), maxc = Scalar(-1000);
  Index minrow=0,mincol=0,maxrow=0,maxcol=0;
  for(Index j = 0; j < cols; j++)
  for(Index i = 0; i < rows; i++)
  {
    if(m(i,j) < minc)
    {
      minc = m(i,j);
      minrow = i;
      mincol = j;
    }
    if(m(i,j) > maxc)
    {
      maxc = m(i,j);
      maxrow = i;
      maxcol = j;
    }
  }
  Index eigen_minrow, eigen_mincol, eigen_maxrow, eigen_maxcol;
  Scalar eigen_minc, eigen_maxc;
  eigen_minc = m.minCoeff(&eigen_minrow,&eigen_mincol);
  eigen_maxc = m.maxCoeff(&eigen_maxrow,&eigen_maxcol);
  VERIFY(minrow == eigen_minrow);
  VERIFY(maxrow == eigen_maxrow);
  VERIFY(mincol == eigen_mincol);
  VERIFY(maxcol == eigen_maxcol);
  VERIFY_IS_APPROX(minc, eigen_minc);
  VERIFY_IS_APPROX(maxc, eigen_maxc);
  VERIFY_IS_APPROX(minc, m.minCoeff());
  VERIFY_IS_APPROX(maxc, m.maxCoeff());

  eigen_maxc = (m.adjoint()*m).maxCoeff(&eigen_maxrow,&eigen_maxcol);
  eigen_maxc = (m.adjoint()*m).eval().maxCoeff(&maxrow,&maxcol);
  VERIFY(maxrow == eigen_maxrow);
  VERIFY(maxcol == eigen_maxcol);
}

template<typename VectorType> void vectorVisitor(const VectorType& w)
{
  typedef typename VectorType::Scalar Scalar;

  Index size = w.size();

  // construct a random vector where all coefficients are different
  VectorType v;
  v = VectorType::Random(size);
  for(Index i = 0; i < size; i++)
    for(Index i2 = 0; i2 < i; i2++)
      while(v(i) == v(i2)) // yes, ==
        v(i) = internal::random<Scalar>();
  
  Scalar minc = v(0), maxc = v(0);
  Index minidx=0, maxidx=0;
  for(Index i = 0; i < size; i++)
  {
    if(v(i) < minc)
    {
      minc = v(i);
      minidx = i;
    }
    if(v(i) > maxc)
    {
      maxc = v(i);
      maxidx = i;
    }
  }
  Index eigen_minidx, eigen_maxidx;
  Scalar eigen_minc, eigen_maxc;
  eigen_minc = v.minCoeff(&eigen_minidx);
  eigen_maxc = v.maxCoeff(&eigen_maxidx);
  VERIFY(minidx == eigen_minidx);
  VERIFY(maxidx == eigen_maxidx);
  VERIFY_IS_APPROX(minc, eigen_minc);
  VERIFY_IS_APPROX(maxc, eigen_maxc);
  VERIFY_IS_APPROX(minc, v.minCoeff());
  VERIFY_IS_APPROX(maxc, v.maxCoeff());
  
  Index idx0 = internal::random<Index>(0,size-1);
  Index idx1 = eigen_minidx;
  Index idx2 = eigen_maxidx;
  VectorType v1(v), v2(v);
  v1(idx0) = v1(idx1);
  v2(idx0) = v2(idx2);
  v1.minCoeff(&eigen_minidx);
  v2.maxCoeff(&eigen_maxidx);
  VERIFY(eigen_minidx == (std::min)(idx0,idx1));
  VERIFY(eigen_maxidx == (std::min)(idx0,idx2));
}

void test_visitor()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( matrixVisitor(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( matrixVisitor(Matrix2f()) );
    CALL_SUBTEST_3( matrixVisitor(Matrix4d()) );
    CALL_SUBTEST_4( matrixVisitor(MatrixXd(8, 12)) );
    CALL_SUBTEST_5( matrixVisitor(Matrix<double,Dynamic,Dynamic,RowMajor>(20, 20)) );
    CALL_SUBTEST_6( matrixVisitor(MatrixXi(8, 12)) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_7( vectorVisitor(Vector4f()) );
    CALL_SUBTEST_7( vectorVisitor(Matrix<int,12,1>()) );
    CALL_SUBTEST_8( vectorVisitor(VectorXd(10)) );
    CALL_SUBTEST_9( vectorVisitor(RowVectorXd(10)) );
    CALL_SUBTEST_10( vectorVisitor(VectorXf(33)) );
  }
}
