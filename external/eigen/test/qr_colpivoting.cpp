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
#include <Eigen/SVD>

template <typename MatrixType>
void cod() {
  Index rows = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index cols2 = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  Index rank = internal::random<Index>(1, (std::min)(rows, cols) - 1);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime,
                 MatrixType::RowsAtCompileTime>
      MatrixQType;
  MatrixType matrix;
  createRandomPIMatrixOfRank(rank, rows, cols, matrix);
  CompleteOrthogonalDecomposition<MatrixType> cod(matrix);
  VERIFY(rank == cod.rank());
  VERIFY(cols - cod.rank() == cod.dimensionOfKernel());
  VERIFY(!cod.isInjective());
  VERIFY(!cod.isInvertible());
  VERIFY(!cod.isSurjective());

  MatrixQType q = cod.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType z = cod.matrixZ();
  VERIFY_IS_UNITARY(z);

  MatrixType t;
  t.setZero(rows, cols);
  t.topLeftCorner(rank, rank) =
      cod.matrixT().topLeftCorner(rank, rank).template triangularView<Upper>();

  MatrixType c = q * t * z * cod.colsPermutation().inverse();
  VERIFY_IS_APPROX(matrix, c);

  MatrixType exact_solution = MatrixType::Random(cols, cols2);
  MatrixType rhs = matrix * exact_solution;
  MatrixType cod_solution = cod.solve(rhs);
  VERIFY_IS_APPROX(rhs, matrix * cod_solution);

  // Verify that we get the same minimum-norm solution as the SVD.
  JacobiSVD<MatrixType> svd(matrix, ComputeThinU | ComputeThinV);
  MatrixType svd_solution = svd.solve(rhs);
  VERIFY_IS_APPROX(cod_solution, svd_solution);

  MatrixType pinv = cod.pseudoInverse();
  VERIFY_IS_APPROX(cod_solution, pinv * rhs);
}

template <typename MatrixType, int Cols2>
void cod_fixedsize() {
  enum {
    Rows = MatrixType::RowsAtCompileTime,
    Cols = MatrixType::ColsAtCompileTime
  };
  typedef typename MatrixType::Scalar Scalar;
  int rank = internal::random<int>(1, (std::min)(int(Rows), int(Cols)) - 1);
  Matrix<Scalar, Rows, Cols> matrix;
  createRandomPIMatrixOfRank(rank, Rows, Cols, matrix);
  CompleteOrthogonalDecomposition<Matrix<Scalar, Rows, Cols> > cod(matrix);
  VERIFY(rank == cod.rank());
  VERIFY(Cols - cod.rank() == cod.dimensionOfKernel());
  VERIFY(cod.isInjective() == (rank == Rows));
  VERIFY(cod.isSurjective() == (rank == Cols));
  VERIFY(cod.isInvertible() == (cod.isInjective() && cod.isSurjective()));

  Matrix<Scalar, Cols, Cols2> exact_solution;
  exact_solution.setRandom(Cols, Cols2);
  Matrix<Scalar, Rows, Cols2> rhs = matrix * exact_solution;
  Matrix<Scalar, Cols, Cols2> cod_solution = cod.solve(rhs);
  VERIFY_IS_APPROX(rhs, matrix * cod_solution);

  // Verify that we get the same minimum-norm solution as the SVD.
  JacobiSVD<MatrixType> svd(matrix, ComputeFullU | ComputeFullV);
  Matrix<Scalar, Cols, Cols2> svd_solution = svd.solve(rhs);
  VERIFY_IS_APPROX(cod_solution, svd_solution);
}

template<typename MatrixType> void qr()
{
  using std::sqrt;

  Index rows = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE), cols = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE), cols2 = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE);
  Index rank = internal::random<Index>(1, (std::min)(rows, cols)-1);

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixQType;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank,rows,cols,m1);
  ColPivHouseholderQR<MatrixType> qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixQType q = qr.householderQ();
  VERIFY_IS_UNITARY(q);

  MatrixType r = qr.matrixQR().template triangularView<Upper>();
  MatrixType c = q * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, c);

  // Verify that the absolute value of the diagonal elements in R are
  // non-increasing until they reach the singularity threshold.
  RealScalar threshold =
      sqrt(RealScalar(rows)) * numext::abs(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(rows, cols) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(rows, cols); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << rank
                << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }

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

template<typename MatrixType, int Cols2> void qr_fixedsize()
{
  using std::sqrt;
  using std::abs;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  int rank = internal::random<int>(1, (std::min)(int(Rows), int(Cols))-1);
  Matrix<Scalar,Rows,Cols> m1;
  createRandomPIMatrixOfRank(rank,Rows,Cols,m1);
  ColPivHouseholderQR<Matrix<Scalar,Rows,Cols> > qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(Cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY_IS_EQUAL(qr.isInjective(), (rank == Rows));
  VERIFY_IS_EQUAL(qr.isSurjective(), (rank == Cols));
  VERIFY_IS_EQUAL(qr.isInvertible(), (qr.isInjective() && qr.isSurjective()));

  Matrix<Scalar,Rows,Cols> r = qr.matrixQR().template triangularView<Upper>();
  Matrix<Scalar,Rows,Cols> c = qr.householderQ() * r * qr.colsPermutation().inverse();
  VERIFY_IS_APPROX(m1, c);

  Matrix<Scalar,Cols,Cols2> m2 = Matrix<Scalar,Cols,Cols2>::Random(Cols,Cols2);
  Matrix<Scalar,Rows,Cols2> m3 = m1*m2;
  m2 = Matrix<Scalar,Cols,Cols2>::Random(Cols,Cols2);
  m2 = qr.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);
  // Verify that the absolute value of the diagonal elements in R are
  // non-increasing until they reache the singularity threshold.
  RealScalar threshold =
      sqrt(RealScalar(Rows)) * (std::abs)(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(int(Rows), int(Cols)) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(int(Rows), int(Cols)); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << rank
                << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }
}

// This test is meant to verify that pivots are chosen such that
// even for a graded matrix, the diagonal of R falls of roughly
// monotonically until it reaches the threshold for singularity.
// We use the so-called Kahan matrix, which is a famous counter-example
// for rank-revealing QR. See
// http://www.netlib.org/lapack/lawnspdf/lawn176.pdf
// page 3 for more detail.
template<typename MatrixType> void qr_kahan_matrix()
{
  using std::sqrt;
  using std::abs;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = 300, cols = rows;

  MatrixType m1;
  m1.setZero(rows,cols);
  RealScalar s = std::pow(NumTraits<RealScalar>::epsilon(), 1.0 / rows);
  RealScalar c = std::sqrt(1 - s*s);
  RealScalar pow_s_i(1.0); // pow(s,i)
  for (Index i = 0; i < rows; ++i) {
    m1(i, i) = pow_s_i;
    m1.row(i).tail(rows - i - 1) = -pow_s_i * c * MatrixType::Ones(1, rows - i - 1);
    pow_s_i *= s;
  }
  m1 = (m1 + m1.transpose()).eval();
  ColPivHouseholderQR<MatrixType> qr(m1);
  MatrixType r = qr.matrixQR().template triangularView<Upper>();

  RealScalar threshold =
      std::sqrt(RealScalar(rows)) * numext::abs(r(0, 0)) * NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < (std::min)(rows, cols) - 1; ++i) {
    RealScalar x = numext::abs(r(i, i));
    RealScalar y = numext::abs(r(i + 1, i + 1));
    if (x < threshold && y < threshold) continue;
    if (!test_isApproxOrLessThan(y, x)) {
      for (Index j = 0; j < (std::min)(rows, cols); ++j) {
        std::cout << "i = " << j << ", |r_ii| = " << numext::abs(r(j, j)) << std::endl;
      }
      std::cout << "Failure at i=" << i << ", rank=" << qr.rank()
                << ", threshold=" << threshold << std::endl;
    }
    VERIFY_IS_APPROX_OR_LESS_THAN(y, x);
  }
}

template<typename MatrixType> void qr_invertible()
{
  using std::log;
  using std::abs;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  int size = internal::random<int>(10,50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (internal::is_same<RealScalar,float>::value)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  ColPivHouseholderQR<MatrixType> qr(m1);
  m3 = MatrixType::Random(size,size);
  m2 = qr.solve(m3);
  //VERIFY_IS_APPROX(m3, m1*m2);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  for(int i = 0; i < size; i++) m1(i,i) = internal::random<Scalar>();
  RealScalar absdet = abs(m1.diagonal().prod());
  m3 = qr.householderQ(); // get a unitary
  m1 = m3 * m1 * m3;
  qr.compute(m1);
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
}

template<typename MatrixType> void qr_verify_assert()
{
  MatrixType tmp;

  ColPivHouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.householderQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
}

void test_qr_colpivoting()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( qr<MatrixXf>() );
    CALL_SUBTEST_2( qr<MatrixXd>() );
    CALL_SUBTEST_3( qr<MatrixXcd>() );
    CALL_SUBTEST_4(( qr_fixedsize<Matrix<float,3,5>, 4 >() ));
    CALL_SUBTEST_5(( qr_fixedsize<Matrix<double,6,2>, 3 >() ));
    CALL_SUBTEST_5(( qr_fixedsize<Matrix<double,1,1>, 1 >() ));
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( cod<MatrixXf>() );
    CALL_SUBTEST_2( cod<MatrixXd>() );
    CALL_SUBTEST_3( cod<MatrixXcd>() );
    CALL_SUBTEST_4(( cod_fixedsize<Matrix<float,3,5>, 4 >() ));
    CALL_SUBTEST_5(( cod_fixedsize<Matrix<double,6,2>, 3 >() ));
    CALL_SUBTEST_5(( cod_fixedsize<Matrix<double,1,1>, 1 >() ));
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( qr_invertible<MatrixXf>() );
    CALL_SUBTEST_2( qr_invertible<MatrixXd>() );
    CALL_SUBTEST_6( qr_invertible<MatrixXcf>() );
    CALL_SUBTEST_3( qr_invertible<MatrixXcd>() );
  }

  CALL_SUBTEST_7(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST_8(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST_1(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST_2(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST_6(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST_3(qr_verify_assert<MatrixXcd>());

  // Test problem size constructors
  CALL_SUBTEST_9(ColPivHouseholderQR<MatrixXf>(10, 20));

  CALL_SUBTEST_1( qr_kahan_matrix<MatrixXf>() );
  CALL_SUBTEST_2( qr_kahan_matrix<MatrixXd>() );
}
