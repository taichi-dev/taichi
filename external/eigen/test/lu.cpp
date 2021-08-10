// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/LU>
using namespace std;

template<typename MatrixType>
typename MatrixType::RealScalar matrix_l1_norm(const MatrixType& m) {
  return m.cwiseAbs().colwise().sum().maxCoeff();
}

template<typename MatrixType> void lu_non_invertible()
{
  typedef typename MatrixType::RealScalar RealScalar;
  /* this test covers the following files:
     LU.h
  */
  Index rows, cols, cols2;
  if(MatrixType::RowsAtCompileTime==Dynamic)
  {
    rows = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE);
  }
  else
  {
    rows = MatrixType::RowsAtCompileTime;
  }
  if(MatrixType::ColsAtCompileTime==Dynamic)
  {
    cols = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE);
    cols2 = internal::random<int>(2,EIGEN_TEST_MAX_SIZE);
  }
  else
  {
    cols2 = cols = MatrixType::ColsAtCompileTime;
  }

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };
  typedef typename internal::kernel_retval_base<FullPivLU<MatrixType> >::ReturnType KernelMatrixType;
  typedef typename internal::image_retval_base<FullPivLU<MatrixType> >::ReturnType ImageMatrixType;
  typedef Matrix<typename MatrixType::Scalar, ColsAtCompileTime, ColsAtCompileTime>
          CMatrixType;
  typedef Matrix<typename MatrixType::Scalar, RowsAtCompileTime, RowsAtCompileTime>
          RMatrixType;

  Index rank = internal::random<Index>(1, (std::min)(rows, cols)-1);

  // The image of the zero matrix should consist of a single (zero) column vector
  VERIFY((MatrixType::Zero(rows,cols).fullPivLu().image(MatrixType::Zero(rows,cols)).cols() == 1));

  // The kernel of the zero matrix is the entire space, and thus is an invertible matrix of dimensions cols.
  KernelMatrixType kernel = MatrixType::Zero(rows,cols).fullPivLu().kernel();
  VERIFY((kernel.fullPivLu().isInvertible()));

  MatrixType m1(rows, cols), m3(rows, cols2);
  CMatrixType m2(cols, cols2);
  createRandomPIMatrixOfRank(rank, rows, cols, m1);

  FullPivLU<MatrixType> lu;

  // The special value 0.01 below works well in tests. Keep in mind that we're only computing the rank
  // of singular values are either 0 or 1.
  // So it's not clear at all that the epsilon should play any role there.
  lu.setThreshold(RealScalar(0.01));
  lu.compute(m1);

  MatrixType u(rows,cols);
  u = lu.matrixLU().template triangularView<Upper>();
  RMatrixType l = RMatrixType::Identity(rows,rows);
  l.block(0,0,rows,(std::min)(rows,cols)).template triangularView<StrictlyLower>()
    = lu.matrixLU().block(0,0,rows,(std::min)(rows,cols));

  VERIFY_IS_APPROX(lu.permutationP() * m1 * lu.permutationQ(), l*u);

  KernelMatrixType m1kernel = lu.kernel();
  ImageMatrixType m1image = lu.image(m1);

  VERIFY_IS_APPROX(m1, lu.reconstructedMatrix());
  VERIFY(rank == lu.rank());
  VERIFY(cols - lu.rank() == lu.dimensionOfKernel());
  VERIFY(!lu.isInjective());
  VERIFY(!lu.isInvertible());
  VERIFY(!lu.isSurjective());
  VERIFY_IS_MUCH_SMALLER_THAN((m1 * m1kernel), m1);
  VERIFY(m1image.fullPivLu().rank() == rank);
  VERIFY_IS_APPROX(m1 * m1.adjoint() * m1image, m1image);

  m2 = CMatrixType::Random(cols,cols2);
  m3 = m1*m2;
  m2 = CMatrixType::Random(cols,cols2);
  // test that the code, which does resize(), may be applied to an xpr
  m2.block(0,0,m2.rows(),m2.cols()) = lu.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);

  // test solve with transposed
  m3 = MatrixType::Random(rows,cols2);
  m2 = m1.transpose()*m3;
  m3 = MatrixType::Random(rows,cols2);
  lu.template _solve_impl_transposed<false>(m2, m3);
  VERIFY_IS_APPROX(m2, m1.transpose()*m3);
  m3 = MatrixType::Random(rows,cols2);
  m3 = lu.transpose().solve(m2);
  VERIFY_IS_APPROX(m2, m1.transpose()*m3);

  // test solve with conjugate transposed
  m3 = MatrixType::Random(rows,cols2);
  m2 = m1.adjoint()*m3;
  m3 = MatrixType::Random(rows,cols2);
  lu.template _solve_impl_transposed<true>(m2, m3);
  VERIFY_IS_APPROX(m2, m1.adjoint()*m3);
  m3 = MatrixType::Random(rows,cols2);
  m3 = lu.adjoint().solve(m2);
  VERIFY_IS_APPROX(m2, m1.adjoint()*m3);
}

template<typename MatrixType> void lu_invertible()
{
  /* this test covers the following files:
     LU.h
  */
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  Index size = MatrixType::RowsAtCompileTime;
  if( size==Dynamic)
    size = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  FullPivLU<MatrixType> lu;
  lu.setThreshold(RealScalar(0.01));
  do {
    m1 = MatrixType::Random(size,size);
    lu.compute(m1);
  } while(!lu.isInvertible());

  VERIFY_IS_APPROX(m1, lu.reconstructedMatrix());
  VERIFY(0 == lu.dimensionOfKernel());
  VERIFY(lu.kernel().cols() == 1); // the kernel() should consist of a single (zero) column vector
  VERIFY(size == lu.rank());
  VERIFY(lu.isInjective());
  VERIFY(lu.isSurjective());
  VERIFY(lu.isInvertible());
  VERIFY(lu.image(m1).fullPivLu().isInvertible());
  m3 = MatrixType::Random(size,size);
  m2 = lu.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);
  MatrixType m1_inverse = lu.inverse();
  VERIFY_IS_APPROX(m2, m1_inverse*m3);

  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m1)) / matrix_l1_norm(m1_inverse);
  const RealScalar rcond_est = lu.rcond();
  // Verify that the estimated condition number is within a factor of 10 of the
  // truth.
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);

  // test solve with transposed
  lu.template _solve_impl_transposed<false>(m3, m2);
  VERIFY_IS_APPROX(m3, m1.transpose()*m2);
  m3 = MatrixType::Random(size,size);
  m3 = lu.transpose().solve(m2);
  VERIFY_IS_APPROX(m2, m1.transpose()*m3);

  // test solve with conjugate transposed
  lu.template _solve_impl_transposed<true>(m3, m2);
  VERIFY_IS_APPROX(m3, m1.adjoint()*m2);
  m3 = MatrixType::Random(size,size);
  m3 = lu.adjoint().solve(m2);
  VERIFY_IS_APPROX(m2, m1.adjoint()*m3);

  // Regression test for Bug 302
  MatrixType m4 = MatrixType::Random(size,size);
  VERIFY_IS_APPROX(lu.solve(m3*m4), lu.solve(m3)*m4);
}

template<typename MatrixType> void lu_partial_piv()
{
  /* this test covers the following files:
     PartialPivLU.h
  */
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  Index size = internal::random<Index>(1,4);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1.setRandom();
  PartialPivLU<MatrixType> plu(m1);

  VERIFY_IS_APPROX(m1, plu.reconstructedMatrix());

  m3 = MatrixType::Random(size,size);
  m2 = plu.solve(m3);
  VERIFY_IS_APPROX(m3, m1*m2);
  MatrixType m1_inverse = plu.inverse();
  VERIFY_IS_APPROX(m2, m1_inverse*m3);

  RealScalar rcond = (RealScalar(1) / matrix_l1_norm(m1)) / matrix_l1_norm(m1_inverse);
  const RealScalar rcond_est = plu.rcond();
  // Verify that the estimate is within a factor of 10 of the truth.
  VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);

  // test solve with transposed
  plu.template _solve_impl_transposed<false>(m3, m2);
  VERIFY_IS_APPROX(m3, m1.transpose()*m2);
  m3 = MatrixType::Random(size,size);
  m3 = plu.transpose().solve(m2);
  VERIFY_IS_APPROX(m2, m1.transpose()*m3);

  // test solve with conjugate transposed
  plu.template _solve_impl_transposed<true>(m3, m2);
  VERIFY_IS_APPROX(m3, m1.adjoint()*m2);
  m3 = MatrixType::Random(size,size);
  m3 = plu.adjoint().solve(m2);
  VERIFY_IS_APPROX(m2, m1.adjoint()*m3);
}

template<typename MatrixType> void lu_verify_assert()
{
  MatrixType tmp;

  FullPivLU<MatrixType> lu;
  VERIFY_RAISES_ASSERT(lu.matrixLU())
  VERIFY_RAISES_ASSERT(lu.permutationP())
  VERIFY_RAISES_ASSERT(lu.permutationQ())
  VERIFY_RAISES_ASSERT(lu.kernel())
  VERIFY_RAISES_ASSERT(lu.image(tmp))
  VERIFY_RAISES_ASSERT(lu.solve(tmp))
  VERIFY_RAISES_ASSERT(lu.determinant())
  VERIFY_RAISES_ASSERT(lu.rank())
  VERIFY_RAISES_ASSERT(lu.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(lu.isInjective())
  VERIFY_RAISES_ASSERT(lu.isSurjective())
  VERIFY_RAISES_ASSERT(lu.isInvertible())
  VERIFY_RAISES_ASSERT(lu.inverse())

  PartialPivLU<MatrixType> plu;
  VERIFY_RAISES_ASSERT(plu.matrixLU())
  VERIFY_RAISES_ASSERT(plu.permutationP())
  VERIFY_RAISES_ASSERT(plu.solve(tmp))
  VERIFY_RAISES_ASSERT(plu.determinant())
  VERIFY_RAISES_ASSERT(plu.inverse())
}

void test_lu()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( lu_non_invertible<Matrix3f>() );
    CALL_SUBTEST_1( lu_invertible<Matrix3f>() );
    CALL_SUBTEST_1( lu_verify_assert<Matrix3f>() );

    CALL_SUBTEST_2( (lu_non_invertible<Matrix<double, 4, 6> >()) );
    CALL_SUBTEST_2( (lu_verify_assert<Matrix<double, 4, 6> >()) );

    CALL_SUBTEST_3( lu_non_invertible<MatrixXf>() );
    CALL_SUBTEST_3( lu_invertible<MatrixXf>() );
    CALL_SUBTEST_3( lu_verify_assert<MatrixXf>() );

    CALL_SUBTEST_4( lu_non_invertible<MatrixXd>() );
    CALL_SUBTEST_4( lu_invertible<MatrixXd>() );
    CALL_SUBTEST_4( lu_partial_piv<MatrixXd>() );
    CALL_SUBTEST_4( lu_verify_assert<MatrixXd>() );

    CALL_SUBTEST_5( lu_non_invertible<MatrixXcf>() );
    CALL_SUBTEST_5( lu_invertible<MatrixXcf>() );
    CALL_SUBTEST_5( lu_verify_assert<MatrixXcf>() );

    CALL_SUBTEST_6( lu_non_invertible<MatrixXcd>() );
    CALL_SUBTEST_6( lu_invertible<MatrixXcd>() );
    CALL_SUBTEST_6( lu_partial_piv<MatrixXcd>() );
    CALL_SUBTEST_6( lu_verify_assert<MatrixXcd>() );

    CALL_SUBTEST_7(( lu_non_invertible<Matrix<float,Dynamic,16> >() ));

    // Test problem size constructors
    CALL_SUBTEST_9( PartialPivLU<MatrixXf>(10) );
    CALL_SUBTEST_9( FullPivLU<MatrixXf>(10, 20); );
  }
}
