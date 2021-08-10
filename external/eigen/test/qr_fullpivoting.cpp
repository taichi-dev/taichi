// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>

template<typename MatrixType> void qr()
{
  Index max_size = EIGEN_TEST_MAX_SIZE;
  Index min_size = numext::maxi(1,EIGEN_TEST_MAX_SIZE/10);
  Index rows  = internal::random<Index>(min_size,max_size),
        cols  = internal::random<Index>(min_size,max_size),
        cols2 = internal::random<Index>(min_size,max_size),
        rank  = internal::random<Index>(1, (std::min)(rows, cols)-1);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank,rows,cols,m1);
  FullPivHouseholderQR<MatrixType> qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixType r = qr.matrixQR();
  
  MatrixQType q = qr.matrixQ();
  VERIFY_IS_UNITARY(q);
  
  // FIXME need better way to construct trapezoid
  for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) if(i>j) r(i,j) = Scalar(0);

  MatrixType c = qr.matrixQ() * r * qr.colsPermutation().inverse();

  VERIFY_IS_APPROX(m1, c);
  
  // stress the ReturnByValue mechanism
  MatrixType tmp;
  VERIFY_IS_APPROX(tmp.noalias() = qr.matrixQ() * r, (qr.matrixQ() * r).eval());
  
  MatrixType m2 = MatrixType::Random(cols,cols2);
  MatrixType m3 = m1*m2;
  m2 = MatrixType::Random(cols,cols2);
  m2 = qr.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);

  {
    Index size = rows;
    do {
      m1 = MatrixType::Random(size,size);
      qr.compute(m1);
    } while(!qr.isInvertible());
    MatrixType m1_inv = qr.inverse();
    m3 = m1 * MatrixType::Random(size,cols2);
    m2 = qr.solve(m3);
    VERIFY_IS_APPROX(m2, m1_inv*m3);
  }
}

template<typename MatrixType> void qr_invertible()
{
  using std::log;
  using std::abs;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  Index max_size = numext::mini(50,EIGEN_TEST_MAX_SIZE);
  Index min_size = numext::maxi(1,EIGEN_TEST_MAX_SIZE/10);
  Index size = internal::random<Index>(min_size,max_size);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (internal::is_same<RealScalar,float>::value)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  FullPivHouseholderQR<MatrixType> qr(m1);
  VERIFY(qr.isInjective());
  VERIFY(qr.isInvertible());
  VERIFY(qr.isSurjective());

  m3 = MatrixType::Random(size,size);
  m2 = qr.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  for(int i = 0; i < size; i++) m1(i,i) = internal::random<Scalar>();
  RealScalar absdet = abs(m1.diagonal().prod());
  m3 = qr.matrixQ(); // get a unitary
  m1 = m3 * m1 * m3;
  qr.compute(m1);
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
}

template<typename MatrixType> void qr_verify_assert()
{
  MatrixType tmp;

  FullPivHouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.matrixQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
}

void test_qr_fullpivoting()
{
 for(int i = 0; i < 1; i++) {
    // FIXME : very weird bug here
//     CALL_SUBTEST(qr(Matrix2f()) );
    CALL_SUBTEST_1( qr<MatrixXf>() );
    CALL_SUBTEST_2( qr<MatrixXd>() );
    CALL_SUBTEST_3( qr<MatrixXcd>() );
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( qr_invertible<MatrixXf>() );
    CALL_SUBTEST_2( qr_invertible<MatrixXd>() );
    CALL_SUBTEST_4( qr_invertible<MatrixXcf>() );
    CALL_SUBTEST_3( qr_invertible<MatrixXcd>() );
  }

  CALL_SUBTEST_5(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST_6(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST_1(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST_2(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST_4(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST_3(qr_verify_assert<MatrixXcd>());

  // Test problem size constructors
  CALL_SUBTEST_7(FullPivHouseholderQR<MatrixXf>(10, 20));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float,10,20> >(10,20)));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float,10,20> >(Matrix<float,10,20>::Random())));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float,20,10> >(20,10)));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float,20,10> >(Matrix<float,20,10>::Random())));
}
