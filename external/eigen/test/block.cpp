// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT // otherwise we fail at compile time on unused paths
#include "main.h"

template<typename MatrixType, typename Index, typename Scalar>
typename Eigen::internal::enable_if<!NumTraits<typename MatrixType::Scalar>::IsComplex,typename MatrixType::Scalar>::type
block_real_only(const MatrixType &m1, Index r1, Index r2, Index c1, Index c2, const Scalar& s1) {
  // check cwise-Functions:
  VERIFY_IS_APPROX(m1.row(r1).cwiseMax(s1), m1.cwiseMax(s1).row(r1));
  VERIFY_IS_APPROX(m1.col(c1).cwiseMin(s1), m1.cwiseMin(s1).col(c1));

  VERIFY_IS_APPROX(m1.block(r1,c1,r2-r1+1,c2-c1+1).cwiseMin(s1), m1.cwiseMin(s1).block(r1,c1,r2-r1+1,c2-c1+1));
  VERIFY_IS_APPROX(m1.block(r1,c1,r2-r1+1,c2-c1+1).cwiseMax(s1), m1.cwiseMax(s1).block(r1,c1,r2-r1+1,c2-c1+1));
  
  return Scalar(0);
}

template<typename MatrixType, typename Index, typename Scalar>
typename Eigen::internal::enable_if<NumTraits<typename MatrixType::Scalar>::IsComplex,typename MatrixType::Scalar>::type
block_real_only(const MatrixType &, Index, Index, Index, Index, const Scalar&) {
  return Scalar(0);
}


template<typename MatrixType> void block(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, MatrixType::IsRowMajor?RowMajor:ColMajor> DynamicMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> DynamicVectorType;
  
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m1_copy = m1,
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             ones = MatrixType::Ones(rows, cols);
  VectorType v1 = VectorType::Random(rows);

  Scalar s1 = internal::random<Scalar>();

  Index r1 = internal::random<Index>(0,rows-1);
  Index r2 = internal::random<Index>(r1,rows-1);
  Index c1 = internal::random<Index>(0,cols-1);
  Index c2 = internal::random<Index>(c1,cols-1);

  block_real_only(m1, r1, r2, c1, c1, s1);

  //check row() and col()
  VERIFY_IS_EQUAL(m1.col(c1).transpose(), m1.transpose().row(c1));
  //check operator(), both constant and non-constant, on row() and col()
  m1 = m1_copy;
  m1.row(r1) += s1 * m1_copy.row(r2);
  VERIFY_IS_APPROX(m1.row(r1), m1_copy.row(r1) + s1 * m1_copy.row(r2));
  // check nested block xpr on lhs
  m1.row(r1).row(0) += s1 * m1_copy.row(r2);
  VERIFY_IS_APPROX(m1.row(r1), m1_copy.row(r1) + Scalar(2) * s1 * m1_copy.row(r2));
  m1 = m1_copy;
  m1.col(c1) += s1 * m1_copy.col(c2);
  VERIFY_IS_APPROX(m1.col(c1), m1_copy.col(c1) + s1 * m1_copy.col(c2));
  m1.col(c1).col(0) += s1 * m1_copy.col(c2);
  VERIFY_IS_APPROX(m1.col(c1), m1_copy.col(c1) + Scalar(2) * s1 * m1_copy.col(c2));
  
  
  //check block()
  Matrix<Scalar,Dynamic,Dynamic> b1(1,1); b1(0,0) = m1(r1,c1);

  RowVectorType br1(m1.block(r1,0,1,cols));
  VectorType bc1(m1.block(0,c1,rows,1));
  VERIFY_IS_EQUAL(b1, m1.block(r1,c1,1,1));
  VERIFY_IS_EQUAL(m1.row(r1), br1);
  VERIFY_IS_EQUAL(m1.col(c1), bc1);
  //check operator(), both constant and non-constant, on block()
  m1.block(r1,c1,r2-r1+1,c2-c1+1) = s1 * m2.block(0, 0, r2-r1+1,c2-c1+1);
  m1.block(r1,c1,r2-r1+1,c2-c1+1)(r2-r1,c2-c1) = m2.block(0, 0, r2-r1+1,c2-c1+1)(0,0);

  enum {
    BlockRows = 2,
    BlockCols = 5
  };
  if (rows>=5 && cols>=8)
  {
    // test fixed block() as lvalue
    m1.template block<BlockRows,BlockCols>(1,1) *= s1;
    // test operator() on fixed block() both as constant and non-constant
    m1.template block<BlockRows,BlockCols>(1,1)(0, 3) = m1.template block<2,5>(1,1)(1,2);
    // check that fixed block() and block() agree
    Matrix<Scalar,Dynamic,Dynamic> b = m1.template block<BlockRows,BlockCols>(3,3);
    VERIFY_IS_EQUAL(b, m1.block(3,3,BlockRows,BlockCols));

    // same tests with mixed fixed/dynamic size
    m1.template block<BlockRows,Dynamic>(1,1,BlockRows,BlockCols) *= s1;
    m1.template block<BlockRows,Dynamic>(1,1,BlockRows,BlockCols)(0,3) = m1.template block<2,5>(1,1)(1,2);
    Matrix<Scalar,Dynamic,Dynamic> b2 = m1.template block<Dynamic,BlockCols>(3,3,2,5);
    VERIFY_IS_EQUAL(b2, m1.block(3,3,BlockRows,BlockCols));
  }

  if (rows>2)
  {
    // test sub vectors
    VERIFY_IS_EQUAL(v1.template head<2>(), v1.block(0,0,2,1));
    VERIFY_IS_EQUAL(v1.template head<2>(), v1.head(2));
    VERIFY_IS_EQUAL(v1.template head<2>(), v1.segment(0,2));
    VERIFY_IS_EQUAL(v1.template head<2>(), v1.template segment<2>(0));
    Index i = rows-2;
    VERIFY_IS_EQUAL(v1.template tail<2>(), v1.block(i,0,2,1));
    VERIFY_IS_EQUAL(v1.template tail<2>(), v1.tail(2));
    VERIFY_IS_EQUAL(v1.template tail<2>(), v1.segment(i,2));
    VERIFY_IS_EQUAL(v1.template tail<2>(), v1.template segment<2>(i));
    i = internal::random<Index>(0,rows-2);
    VERIFY_IS_EQUAL(v1.segment(i,2), v1.template segment<2>(i));
  }

  // stress some basic stuffs with block matrices
  VERIFY(numext::real(ones.col(c1).sum()) == RealScalar(rows));
  VERIFY(numext::real(ones.row(r1).sum()) == RealScalar(cols));

  VERIFY(numext::real(ones.col(c1).dot(ones.col(c2))) == RealScalar(rows));
  VERIFY(numext::real(ones.row(r1).dot(ones.row(r2))) == RealScalar(cols));
  
  // check that linear acccessors works on blocks
  m1 = m1_copy;
  if((MatrixType::Flags&RowMajorBit)==0)
    VERIFY_IS_EQUAL(m1.leftCols(c1).coeff(r1+c1*rows), m1(r1,c1));
  else
    VERIFY_IS_EQUAL(m1.topRows(r1).coeff(c1+r1*cols), m1(r1,c1));
  

  // now test some block-inside-of-block.
  
  // expressions with direct access
  VERIFY_IS_EQUAL( (m1.block(r1,c1,rows-r1,cols-c1).block(r2-r1,c2-c1,rows-r2,cols-c2)) , (m1.block(r2,c2,rows-r2,cols-c2)) );
  VERIFY_IS_EQUAL( (m1.block(r1,c1,r2-r1+1,c2-c1+1).row(0)) , (m1.row(r1).segment(c1,c2-c1+1)) );
  VERIFY_IS_EQUAL( (m1.block(r1,c1,r2-r1+1,c2-c1+1).col(0)) , (m1.col(c1).segment(r1,r2-r1+1)) );
  VERIFY_IS_EQUAL( (m1.block(r1,c1,r2-r1+1,c2-c1+1).transpose().col(0)) , (m1.row(r1).segment(c1,c2-c1+1)).transpose() );
  VERIFY_IS_EQUAL( (m1.transpose().block(c1,r1,c2-c1+1,r2-r1+1).col(0)) , (m1.row(r1).segment(c1,c2-c1+1)).transpose() );

  // expressions without direct access
  VERIFY_IS_APPROX( ((m1+m2).block(r1,c1,rows-r1,cols-c1).block(r2-r1,c2-c1,rows-r2,cols-c2)) , ((m1+m2).block(r2,c2,rows-r2,cols-c2)) );
  VERIFY_IS_APPROX( ((m1+m2).block(r1,c1,r2-r1+1,c2-c1+1).row(0)) , ((m1+m2).row(r1).segment(c1,c2-c1+1)) );
  VERIFY_IS_APPROX( ((m1+m2).block(r1,c1,r2-r1+1,c2-c1+1).col(0)) , ((m1+m2).col(c1).segment(r1,r2-r1+1)) );
  VERIFY_IS_APPROX( ((m1+m2).block(r1,c1,r2-r1+1,c2-c1+1).transpose().col(0)) , ((m1+m2).row(r1).segment(c1,c2-c1+1)).transpose() );
  VERIFY_IS_APPROX( ((m1+m2).transpose().block(c1,r1,c2-c1+1,r2-r1+1).col(0)) , ((m1+m2).row(r1).segment(c1,c2-c1+1)).transpose() );

  VERIFY_IS_APPROX( (m1*1).topRows(r1),  m1.topRows(r1) );
  VERIFY_IS_APPROX( (m1*1).leftCols(c1), m1.leftCols(c1) );
  VERIFY_IS_APPROX( (m1*1).transpose().topRows(c1), m1.transpose().topRows(c1) );
  VERIFY_IS_APPROX( (m1*1).transpose().leftCols(r1), m1.transpose().leftCols(r1) );
  VERIFY_IS_APPROX( (m1*1).transpose().middleRows(c1,c2-c1+1), m1.transpose().middleRows(c1,c2-c1+1) );
  VERIFY_IS_APPROX( (m1*1).transpose().middleCols(r1,r2-r1+1), m1.transpose().middleCols(r1,r2-r1+1) );

  // evaluation into plain matrices from expressions with direct access (stress MapBase)
  DynamicMatrixType dm;
  DynamicVectorType dv;
  dm.setZero();
  dm = m1.block(r1,c1,rows-r1,cols-c1).block(r2-r1,c2-c1,rows-r2,cols-c2);
  VERIFY_IS_EQUAL(dm, (m1.block(r2,c2,rows-r2,cols-c2)));
  dm.setZero();
  dv.setZero();
  dm = m1.block(r1,c1,r2-r1+1,c2-c1+1).row(0).transpose();
  dv = m1.row(r1).segment(c1,c2-c1+1);
  VERIFY_IS_EQUAL(dv, dm);
  dm.setZero();
  dv.setZero();
  dm = m1.col(c1).segment(r1,r2-r1+1);
  dv = m1.block(r1,c1,r2-r1+1,c2-c1+1).col(0);
  VERIFY_IS_EQUAL(dv, dm);
  dm.setZero();
  dv.setZero();
  dm = m1.block(r1,c1,r2-r1+1,c2-c1+1).transpose().col(0);
  dv = m1.row(r1).segment(c1,c2-c1+1);
  VERIFY_IS_EQUAL(dv, dm);
  dm.setZero();
  dv.setZero();
  dm = m1.row(r1).segment(c1,c2-c1+1).transpose();
  dv = m1.transpose().block(c1,r1,c2-c1+1,r2-r1+1).col(0);
  VERIFY_IS_EQUAL(dv, dm);

  VERIFY_IS_EQUAL( (m1.template block<Dynamic,1>(1,0,0,1)), m1.block(1,0,0,1));
  VERIFY_IS_EQUAL( (m1.template block<1,Dynamic>(0,1,1,0)), m1.block(0,1,1,0));
  VERIFY_IS_EQUAL( ((m1*1).template block<Dynamic,1>(1,0,0,1)), m1.block(1,0,0,1));
  VERIFY_IS_EQUAL( ((m1*1).template block<1,Dynamic>(0,1,1,0)), m1.block(0,1,1,0));

  if (rows>=2 && cols>=2)
  {
    VERIFY_RAISES_ASSERT( m1 += m1.col(0) );
    VERIFY_RAISES_ASSERT( m1 -= m1.col(0) );
    VERIFY_RAISES_ASSERT( m1.array() *= m1.col(0).array() );
    VERIFY_RAISES_ASSERT( m1.array() /= m1.col(0).array() );
  }
}


template<typename MatrixType>
void compare_using_data_and_stride(const MatrixType& m)
{
  Index rows = m.rows();
  Index cols = m.cols();
  Index size = m.size();
  Index innerStride = m.innerStride();
  Index outerStride = m.outerStride();
  Index rowStride = m.rowStride();
  Index colStride = m.colStride();
  const typename MatrixType::Scalar* data = m.data();

  for(int j=0;j<cols;++j)
    for(int i=0;i<rows;++i)
      VERIFY(m.coeff(i,j) == data[i*rowStride + j*colStride]);

  if(!MatrixType::IsVectorAtCompileTime)
  {
    for(int j=0;j<cols;++j)
      for(int i=0;i<rows;++i)
        VERIFY(m.coeff(i,j) == data[(MatrixType::Flags&RowMajorBit)
                                     ? i*outerStride + j*innerStride
                                     : j*outerStride + i*innerStride]);
  }

  if(MatrixType::IsVectorAtCompileTime)
  {
    VERIFY(innerStride == int((&m.coeff(1))-(&m.coeff(0))));
    for (int i=0;i<size;++i)
      VERIFY(m.coeff(i) == data[i*innerStride]);
  }
}

template<typename MatrixType>
void data_and_stride(const MatrixType& m)
{
  Index rows = m.rows();
  Index cols = m.cols();

  Index r1 = internal::random<Index>(0,rows-1);
  Index r2 = internal::random<Index>(r1,rows-1);
  Index c1 = internal::random<Index>(0,cols-1);
  Index c2 = internal::random<Index>(c1,cols-1);

  MatrixType m1 = MatrixType::Random(rows, cols);
  compare_using_data_and_stride(m1.block(r1, c1, r2-r1+1, c2-c1+1));
  compare_using_data_and_stride(m1.transpose().block(c1, r1, c2-c1+1, r2-r1+1));
  compare_using_data_and_stride(m1.row(r1));
  compare_using_data_and_stride(m1.col(c1));
  compare_using_data_and_stride(m1.row(r1).transpose());
  compare_using_data_and_stride(m1.col(c1).transpose());
}

void test_block()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( block(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( block(Matrix4d()) );
    CALL_SUBTEST_3( block(MatrixXcf(3, 3)) );
    CALL_SUBTEST_4( block(MatrixXi(8, 12)) );
    CALL_SUBTEST_5( block(MatrixXcd(20, 20)) );
    CALL_SUBTEST_6( block(MatrixXf(20, 20)) );

    CALL_SUBTEST_8( block(Matrix<float,Dynamic,4>(3, 4)) );

#ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
    CALL_SUBTEST_6( data_and_stride(MatrixXf(internal::random(5,50), internal::random(5,50))) );
    CALL_SUBTEST_7( data_and_stride(Matrix<int,Dynamic,Dynamic,RowMajor>(internal::random(5,50), internal::random(5,50))) );
#endif
  }
}
