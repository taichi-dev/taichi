// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
// ^^ see bug 1449

#include "main.h"

template<typename MatrixType> void matrixRedux(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  // The entries of m1 are uniformly distributed in [0,1], so m1.prod() is very small. This may lead to test
  // failures if we underflow into denormals. Thus, we scale so that entries are close to 1.
  MatrixType m1_for_prod = MatrixType::Ones(rows, cols) + RealScalar(0.2) * m1;

  VERIFY_IS_MUCH_SMALLER_THAN(MatrixType::Zero(rows, cols).sum(), Scalar(1));
  VERIFY_IS_APPROX(MatrixType::Ones(rows, cols).sum(), Scalar(float(rows*cols))); // the float() here to shut up excessive MSVC warning about int->complex conversion being lossy
  Scalar s(0), p(1), minc(numext::real(m1.coeff(0))), maxc(numext::real(m1.coeff(0)));
  for(int j = 0; j < cols; j++)
  for(int i = 0; i < rows; i++)
  {
    s += m1(i,j);
    p *= m1_for_prod(i,j);
    minc = (std::min)(numext::real(minc), numext::real(m1(i,j)));
    maxc = (std::max)(numext::real(maxc), numext::real(m1(i,j)));
  }
  const Scalar mean = s/Scalar(RealScalar(rows*cols));

  VERIFY_IS_APPROX(m1.sum(), s);
  VERIFY_IS_APPROX(m1.mean(), mean);
  VERIFY_IS_APPROX(m1_for_prod.prod(), p);
  VERIFY_IS_APPROX(m1.real().minCoeff(), numext::real(minc));
  VERIFY_IS_APPROX(m1.real().maxCoeff(), numext::real(maxc));

  // test slice vectorization assuming assign is ok
  Index r0 = internal::random<Index>(0,rows-1);
  Index c0 = internal::random<Index>(0,cols-1);
  Index r1 = internal::random<Index>(r0+1,rows)-r0;
  Index c1 = internal::random<Index>(c0+1,cols)-c0;
  VERIFY_IS_APPROX(m1.block(r0,c0,r1,c1).sum(), m1.block(r0,c0,r1,c1).eval().sum());
  VERIFY_IS_APPROX(m1.block(r0,c0,r1,c1).mean(), m1.block(r0,c0,r1,c1).eval().mean());
  VERIFY_IS_APPROX(m1_for_prod.block(r0,c0,r1,c1).prod(), m1_for_prod.block(r0,c0,r1,c1).eval().prod());
  VERIFY_IS_APPROX(m1.block(r0,c0,r1,c1).real().minCoeff(), m1.block(r0,c0,r1,c1).real().eval().minCoeff());
  VERIFY_IS_APPROX(m1.block(r0,c0,r1,c1).real().maxCoeff(), m1.block(r0,c0,r1,c1).real().eval().maxCoeff());

  // regression for bug 1090
  const int R1 = MatrixType::RowsAtCompileTime>=2 ? MatrixType::RowsAtCompileTime/2 : 6;
  const int C1 = MatrixType::ColsAtCompileTime>=2 ? MatrixType::ColsAtCompileTime/2 : 6;
  if(R1<=rows-r0 && C1<=cols-c0)
  {
    VERIFY_IS_APPROX( (m1.template block<R1,C1>(r0,c0).sum()), m1.block(r0,c0,R1,C1).sum() );
  }
  
  // test empty objects
  VERIFY_IS_APPROX(m1.block(r0,c0,0,0).sum(),   Scalar(0));
  VERIFY_IS_APPROX(m1.block(r0,c0,0,0).prod(),  Scalar(1));

  // test nesting complex expression
  VERIFY_EVALUATION_COUNT( (m1.matrix()*m1.matrix().transpose()).sum(), (MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime!=1 ? 0 : 1) );
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> m2(rows,rows);
  m2.setRandom();
  VERIFY_EVALUATION_COUNT( ((m1.matrix()*m1.matrix().transpose())+m2).sum(),(MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime!=1 ? 0 : 1));
}

template<typename VectorType> void vectorRedux(const VectorType& w)
{
  using std::abs;
  typedef typename VectorType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index size = w.size();

  VectorType v = VectorType::Random(size);
  VectorType v_for_prod = VectorType::Ones(size) + Scalar(0.2) * v; // see comment above declaration of m1_for_prod

  for(int i = 1; i < size; i++)
  {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(0))), maxc(numext::real(v.coeff(0)));
    for(int j = 0; j < i; j++)
    {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.head(i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.head(i).prod());
    VERIFY_IS_APPROX(minc, v.real().head(i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().head(i).maxCoeff());
  }

  for(int i = 0; i < size-1; i++)
  {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for(int j = i; j < size; j++)
    {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.tail(size-i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.tail(size-i).prod());
    VERIFY_IS_APPROX(minc, v.real().tail(size-i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().tail(size-i).maxCoeff());
  }

  for(int i = 0; i < size/2; i++)
  {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for(int j = i; j < size-i; j++)
    {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.segment(i, size-2*i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.segment(i, size-2*i).prod());
    VERIFY_IS_APPROX(minc, v.real().segment(i, size-2*i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().segment(i, size-2*i).maxCoeff());
  }
  
  // test empty objects
  VERIFY_IS_APPROX(v.head(0).sum(),   Scalar(0));
  VERIFY_IS_APPROX(v.tail(0).prod(),  Scalar(1));
  VERIFY_RAISES_ASSERT(v.head(0).mean());
  VERIFY_RAISES_ASSERT(v.head(0).minCoeff());
  VERIFY_RAISES_ASSERT(v.head(0).maxCoeff());
}

void test_redux()
{
  // the max size cannot be too large, otherwise reduxion operations obviously generate large errors.
  int maxsize = (std::min)(100,EIGEN_TEST_MAX_SIZE);
  TEST_SET_BUT_UNUSED_VARIABLE(maxsize);
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( matrixRedux(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( matrixRedux(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( matrixRedux(Matrix2f()) );
    CALL_SUBTEST_2( matrixRedux(Array2f()) );
    CALL_SUBTEST_2( matrixRedux(Array22f()) );
    CALL_SUBTEST_3( matrixRedux(Matrix4d()) );
    CALL_SUBTEST_3( matrixRedux(Array4d()) );
    CALL_SUBTEST_3( matrixRedux(Array44d()) );
    CALL_SUBTEST_4( matrixRedux(MatrixXcf(internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_4( matrixRedux(ArrayXXcf(internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_5( matrixRedux(MatrixXd (internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_5( matrixRedux(ArrayXXd (internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_6( matrixRedux(MatrixXi (internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_6( matrixRedux(ArrayXXi (internal::random<int>(1,maxsize), internal::random<int>(1,maxsize))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_7( vectorRedux(Vector4f()) );
    CALL_SUBTEST_7( vectorRedux(Array4f()) );
    CALL_SUBTEST_5( vectorRedux(VectorXd(internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_5( vectorRedux(ArrayXd(internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_8( vectorRedux(VectorXf(internal::random<int>(1,maxsize))) );
    CALL_SUBTEST_8( vectorRedux(ArrayXf(internal::random<int>(1,maxsize))) );
  }
}
