// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
using namespace std;
template<typename MatrixType> void diagonalmatrices(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef Matrix<Scalar, Rows, 1> VectorType;
  typedef Matrix<Scalar, 1, Cols> RowVectorType;
  typedef Matrix<Scalar, Rows, Rows> SquareMatrixType;
  typedef Matrix<Scalar, Dynamic, Dynamic> DynMatrixType;
  typedef DiagonalMatrix<Scalar, Rows> LeftDiagonalMatrix;
  typedef DiagonalMatrix<Scalar, Cols> RightDiagonalMatrix;
  typedef Matrix<Scalar, Rows==Dynamic?Dynamic:2*Rows, Cols==Dynamic?Dynamic:2*Cols> BigMatrix;
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols),
             rv2 = RowVectorType::Random(cols);

  LeftDiagonalMatrix ldm1(v1), ldm2(v2);
  RightDiagonalMatrix rdm1(rv1), rdm2(rv2);
  
  Scalar s1 = internal::random<Scalar>();

  SquareMatrixType sq_m1 (v1.asDiagonal());
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  sq_m1 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  SquareMatrixType sq_m2 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, sq_m2);
  
  ldm1 = v1.asDiagonal();
  LeftDiagonalMatrix ldm3(v1);
  VERIFY_IS_APPROX(ldm1.diagonal(), ldm3.diagonal());
  LeftDiagonalMatrix ldm4 = v1.asDiagonal();
  VERIFY_IS_APPROX(ldm1.diagonal(), ldm4.diagonal());
  
  sq_m1.block(0,0,rows,rows) = ldm1;
  VERIFY_IS_APPROX(sq_m1, ldm1.toDenseMatrix());
  sq_m1.transpose() = ldm1;
  VERIFY_IS_APPROX(sq_m1, ldm1.toDenseMatrix());
  
  Index i = internal::random<Index>(0, rows-1);
  Index j = internal::random<Index>(0, cols-1);
  
  VERIFY_IS_APPROX( ((ldm1 * m1)(i,j))  , ldm1.diagonal()(i) * m1(i,j) );
  VERIFY_IS_APPROX( ((ldm1 * (m1+m2))(i,j))  , ldm1.diagonal()(i) * (m1+m2)(i,j) );
  VERIFY_IS_APPROX( ((m1 * rdm1)(i,j))  , rdm1.diagonal()(j) * m1(i,j) );
  VERIFY_IS_APPROX( ((v1.asDiagonal() * m1)(i,j))  , v1(i) * m1(i,j) );
  VERIFY_IS_APPROX( ((m1 * rv1.asDiagonal())(i,j))  , rv1(j) * m1(i,j) );
  VERIFY_IS_APPROX( (((v1+v2).asDiagonal() * m1)(i,j))  , (v1+v2)(i) * m1(i,j) );
  VERIFY_IS_APPROX( (((v1+v2).asDiagonal() * (m1+m2))(i,j))  , (v1+v2)(i) * (m1+m2)(i,j) );
  VERIFY_IS_APPROX( ((m1 * (rv1+rv2).asDiagonal())(i,j))  , (rv1+rv2)(j) * m1(i,j) );
  VERIFY_IS_APPROX( (((m1+m2) * (rv1+rv2).asDiagonal())(i,j))  , (rv1+rv2)(j) * (m1+m2)(i,j) );
  
  if(rows>1)
  {
    DynMatrixType tmp = m1.topRows(rows/2), res;
    VERIFY_IS_APPROX( (res = m1.topRows(rows/2) * rv1.asDiagonal()), tmp * rv1.asDiagonal() );
    VERIFY_IS_APPROX( (res = v1.head(rows/2).asDiagonal()*m1.topRows(rows/2)), v1.head(rows/2).asDiagonal()*tmp );
  }

  BigMatrix big;
  big.setZero(2*rows, 2*cols);
  
  big.block(i,j,rows,cols) = m1;
  big.block(i,j,rows,cols) = v1.asDiagonal() * big.block(i,j,rows,cols);
  
  VERIFY_IS_APPROX((big.block(i,j,rows,cols)) , v1.asDiagonal() * m1 );
  
  big.block(i,j,rows,cols) = m1;
  big.block(i,j,rows,cols) = big.block(i,j,rows,cols) * rv1.asDiagonal();
  VERIFY_IS_APPROX((big.block(i,j,rows,cols)) , m1 * rv1.asDiagonal() );
  
  
  // scalar multiple
  VERIFY_IS_APPROX(LeftDiagonalMatrix(ldm1*s1).diagonal(), ldm1.diagonal() * s1);
  VERIFY_IS_APPROX(LeftDiagonalMatrix(s1*ldm1).diagonal(), s1 * ldm1.diagonal());
  
  VERIFY_IS_APPROX(m1 * (rdm1 * s1), (m1 * rdm1) * s1);
  VERIFY_IS_APPROX(m1 * (s1 * rdm1), (m1 * rdm1) * s1);
  
  // Diagonal to dense
  sq_m1.setRandom();
  sq_m2 = sq_m1;
  VERIFY_IS_APPROX( (sq_m1 += (s1*v1).asDiagonal()), sq_m2 += (s1*v1).asDiagonal().toDenseMatrix() );
  VERIFY_IS_APPROX( (sq_m1 -= (s1*v1).asDiagonal()), sq_m2 -= (s1*v1).asDiagonal().toDenseMatrix() );
  VERIFY_IS_APPROX( (sq_m1 = (s1*v1).asDiagonal()), (s1*v1).asDiagonal().toDenseMatrix() );

  sq_m1.setRandom();
  sq_m2 = v1.asDiagonal();
  sq_m2 = sq_m1 * sq_m2;
  VERIFY_IS_APPROX( (sq_m1*v1.asDiagonal()).col(i), sq_m2.col(i) );
  VERIFY_IS_APPROX( (sq_m1*v1.asDiagonal()).row(i), sq_m2.row(i) );
}

template<typename MatrixType> void as_scalar_product(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> DynMatrixType;
  typedef Matrix<Scalar, Dynamic, 1> DynVectorType;
  typedef Matrix<Scalar, 1, Dynamic> DynRowVectorType;

  Index rows = m.rows();
  Index depth = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);

  VectorType v1 = VectorType::Random(rows);  
  DynVectorType     dv1  = DynVectorType::Random(depth);
  DynRowVectorType  drv1 = DynRowVectorType::Random(depth);
  DynMatrixType     dm1  = dv1;
  DynMatrixType     drm1 = drv1;
  
  Scalar s = v1(0);

  VERIFY_IS_APPROX( v1.asDiagonal() * drv1, s*drv1 );
  VERIFY_IS_APPROX( dv1 * v1.asDiagonal(), dv1*s );

  VERIFY_IS_APPROX( v1.asDiagonal() * drm1, s*drm1 );
  VERIFY_IS_APPROX( dm1 * v1.asDiagonal(), dm1*s );
}

template<int>
void bug987()
{
  Matrix3Xd points = Matrix3Xd::Random(3, 3);
  Vector2d diag = Vector2d::Random();
  Matrix2Xd tmp1 = points.topRows<2>(), res1, res2;
  VERIFY_IS_APPROX( res1 = diag.asDiagonal() * points.topRows<2>(), res2 = diag.asDiagonal() * tmp1 );
  Matrix2d tmp2 = points.topLeftCorner<2,2>();
  VERIFY_IS_APPROX(( res1 = points.topLeftCorner<2,2>()*diag.asDiagonal()) , res2 = tmp2*diag.asDiagonal() );
}

void test_diagonalmatrices()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( diagonalmatrices(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( as_scalar_product(Matrix<float, 1, 1>()) );

    CALL_SUBTEST_2( diagonalmatrices(Matrix3f()) );
    CALL_SUBTEST_3( diagonalmatrices(Matrix<double,3,3,RowMajor>()) );
    CALL_SUBTEST_4( diagonalmatrices(Matrix4d()) );
    CALL_SUBTEST_5( diagonalmatrices(Matrix<float,4,4,RowMajor>()) );
    CALL_SUBTEST_6( diagonalmatrices(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( as_scalar_product(MatrixXcf(1,1)) );
    CALL_SUBTEST_7( diagonalmatrices(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_8( diagonalmatrices(Matrix<double,Dynamic,Dynamic,RowMajor>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_9( diagonalmatrices(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_9( diagonalmatrices(MatrixXf(1,1)) );
    CALL_SUBTEST_9( as_scalar_product(MatrixXf(1,1)) );
  }
  CALL_SUBTEST_10( bug987<0>() );
}
