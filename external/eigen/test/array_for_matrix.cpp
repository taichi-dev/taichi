// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void array_for_matrix(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType; 

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  ColVectorType cv1 = ColVectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols);
  
  Scalar  s1 = internal::random<Scalar>(),
          s2 = internal::random<Scalar>();
          
  // scalar addition
  VERIFY_IS_APPROX(m1.array() + s1, s1 + m1.array());
  VERIFY_IS_APPROX((m1.array() + s1).matrix(), MatrixType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX(((m1*Scalar(2)).array() - s2).matrix(), (m1+m1) - MatrixType::Constant(rows,cols,s2) );
  m3 = m1;
  m3.array() += s2;
  VERIFY_IS_APPROX(m3, (m1.array() + s2).matrix());
  m3 = m1;
  m3.array() -= s1;
  VERIFY_IS_APPROX(m3, (m1.array() - s1).matrix());

  // reductions
  VERIFY_IS_MUCH_SMALLER_THAN(m1.colwise().sum().sum() - m1.sum(), m1.squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.rowwise().sum().sum() - m1.sum(), m1.squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.colwise().sum() + m2.colwise().sum() - (m1+m2).colwise().sum(), (m1+m2).squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.rowwise().sum() - m2.rowwise().sum() - (m1-m2).rowwise().sum(), (m1-m2).squaredNorm());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(internal::scalar_sum_op<Scalar,Scalar>()));

  // vector-wise ops
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() += cv1, m1.colwise() + cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() -= cv1, m1.colwise() - cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() += rv1, m1.rowwise() + rv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() -= rv1, m1.rowwise() - rv1);
  
  // empty objects
  VERIFY_IS_APPROX(m1.block(0,0,0,cols).colwise().sum(),  RowVectorType::Zero(cols));
  VERIFY_IS_APPROX(m1.block(0,0,rows,0).rowwise().prod(), ColVectorType::Ones(rows));
  
  // verify the const accessors exist
  const Scalar& ref_m1 = m.matrix().array().coeffRef(0);
  const Scalar& ref_m2 = m.matrix().array().coeffRef(0,0);
  const Scalar& ref_a1 = m.array().matrix().coeffRef(0);
  const Scalar& ref_a2 = m.array().matrix().coeffRef(0,0);
  VERIFY(&ref_a1 == &ref_m1);
  VERIFY(&ref_a2 == &ref_m2);

  // Check write accessors:
  m1.array().coeffRef(0,0) = 1;
  VERIFY_IS_APPROX(m1(0,0),Scalar(1));
  m1.array()(0,0) = 2;
  VERIFY_IS_APPROX(m1(0,0),Scalar(2));
  m1.array().matrix().coeffRef(0,0) = 3;
  VERIFY_IS_APPROX(m1(0,0),Scalar(3));
  m1.array().matrix()(0,0) = 4;
  VERIFY_IS_APPROX(m1(0,0),Scalar(4));
}

template<typename MatrixType> void comparisons(const MatrixType& m)
{
  using std::abs;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  VERIFY(((m1.array() + Scalar(1)) > m1.array()).all());
  VERIFY(((m1.array() - Scalar(1)) < m1.array()).all());
  if (rows*cols>1)
  {
    m3 = m1;
    m3(r,c) += 1;
    VERIFY(! (m1.array() < m3.array()).all() );
    VERIFY(! (m1.array() > m3.array()).all() );
  }

  // comparisons to scalar
  VERIFY( (m1.array() != (m1(r,c)+1) ).any() );
  VERIFY( (m1.array() > (m1(r,c)-1) ).any() );
  VERIFY( (m1.array() < (m1(r,c)+1) ).any() );
  VERIFY( (m1.array() == m1(r,c) ).any() );
  VERIFY( m1.cwiseEqual(m1(r,c)).any() );

  // test Select
  VERIFY_IS_APPROX( (m1.array()<m2.array()).select(m1,m2), m1.cwiseMin(m2) );
  VERIFY_IS_APPROX( (m1.array()>m2.array()).select(m1,m2), m1.cwiseMax(m2) );
  Scalar mid = (m1.cwiseAbs().minCoeff() + m1.cwiseAbs().maxCoeff())/Scalar(2);
  for (int j=0; j<cols; ++j)
  for (int i=0; i<rows; ++i)
    m3(i,j) = abs(m1(i,j))<mid ? 0 : m1(i,j);
  VERIFY_IS_APPROX( (m1.array().abs()<MatrixType::Constant(rows,cols,mid).array())
                        .select(MatrixType::Zero(rows,cols),m1), m3);
  // shorter versions:
  VERIFY_IS_APPROX( (m1.array().abs()<MatrixType::Constant(rows,cols,mid).array())
                        .select(0,m1), m3);
  VERIFY_IS_APPROX( (m1.array().abs()>=MatrixType::Constant(rows,cols,mid).array())
                        .select(m1,0), m3);
  // even shorter version:
  VERIFY_IS_APPROX( (m1.array().abs()<mid).select(0,m1), m3);

  // count
  VERIFY(((m1.array().abs()+1)>RealScalar(0.1)).count() == rows*cols);

  // and/or
  VERIFY( ((m1.array()<RealScalar(0)).matrix() && (m1.array()>RealScalar(0)).matrix()).count() == 0);
  VERIFY( ((m1.array()<RealScalar(0)).matrix() || (m1.array()>=RealScalar(0)).matrix()).count() == rows*cols);
  RealScalar a = m1.cwiseAbs().mean();
  VERIFY( ((m1.array()<-a).matrix() || (m1.array()>a).matrix()).count() == (m1.cwiseAbs().array()>a).count());

  typedef Matrix<typename MatrixType::Index, Dynamic, 1> VectorOfIndices;

  // TODO allows colwise/rowwise for array
  VERIFY_IS_APPROX(((m1.array().abs()+1)>RealScalar(0.1)).matrix().colwise().count(), VectorOfIndices::Constant(cols,rows).transpose());
  VERIFY_IS_APPROX(((m1.array().abs()+1)>RealScalar(0.1)).matrix().rowwise().count(), VectorOfIndices::Constant(rows, cols));
}

template<typename VectorType> void lpNorm(const VectorType& v)
{
  using std::sqrt;
  typedef typename VectorType::RealScalar RealScalar;
  VectorType u = VectorType::Random(v.size());

  if(v.size()==0)
  {
    VERIFY_IS_APPROX(u.template lpNorm<Infinity>(), RealScalar(0));
    VERIFY_IS_APPROX(u.template lpNorm<1>(), RealScalar(0));
    VERIFY_IS_APPROX(u.template lpNorm<2>(), RealScalar(0));
    VERIFY_IS_APPROX(u.template lpNorm<5>(), RealScalar(0));
  }
  else
  {
    VERIFY_IS_APPROX(u.template lpNorm<Infinity>(), u.cwiseAbs().maxCoeff());
  }

  VERIFY_IS_APPROX(u.template lpNorm<1>(), u.cwiseAbs().sum());
  VERIFY_IS_APPROX(u.template lpNorm<2>(), sqrt(u.array().abs().square().sum()));
  VERIFY_IS_APPROX(numext::pow(u.template lpNorm<5>(), typename VectorType::RealScalar(5)), u.array().abs().pow(5).sum());
}

template<typename MatrixType> void cwise_min_max(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  // min/max with array
  Scalar maxM1 = m1.maxCoeff();
  Scalar minM1 = m1.minCoeff();

  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, minM1), m1.cwiseMin(MatrixType::Constant(rows,cols, minM1)));
  VERIFY_IS_APPROX(m1, m1.cwiseMin(MatrixType::Constant(rows,cols, maxM1)));

  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, maxM1), m1.cwiseMax(MatrixType::Constant(rows,cols, maxM1)));
  VERIFY_IS_APPROX(m1, m1.cwiseMax(MatrixType::Constant(rows,cols, minM1)));

  // min/max with scalar input
  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, minM1), m1.cwiseMin( minM1));
  VERIFY_IS_APPROX(m1, m1.cwiseMin(maxM1));
  VERIFY_IS_APPROX(-m1, (-m1).cwiseMin(-minM1));
  VERIFY_IS_APPROX(-m1.array(), ((-m1).array().min)( -minM1));

  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, maxM1), m1.cwiseMax( maxM1));
  VERIFY_IS_APPROX(m1, m1.cwiseMax(minM1));
  VERIFY_IS_APPROX(-m1, (-m1).cwiseMax(-maxM1));
  VERIFY_IS_APPROX(-m1.array(), ((-m1).array().max)(-maxM1));

  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, minM1).array(), (m1.array().min)( minM1));
  VERIFY_IS_APPROX(m1.array(), (m1.array().min)( maxM1));

  VERIFY_IS_APPROX(MatrixType::Constant(rows,cols, maxM1).array(), (m1.array().max)( maxM1));
  VERIFY_IS_APPROX(m1.array(), (m1.array().max)( minM1));

}

template<typename MatrixTraits> void resize(const MatrixTraits& t)
{
  typedef typename MatrixTraits::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
  typedef Array<Scalar,Dynamic,Dynamic> Array2DType;
  typedef Matrix<Scalar,Dynamic,1> VectorType;
  typedef Array<Scalar,Dynamic,1> Array1DType;

  Index rows = t.rows(), cols = t.cols();

  MatrixType m(rows,cols);
  VectorType v(rows);
  Array2DType a2(rows,cols);
  Array1DType a1(rows);

  m.array().resize(rows+1,cols+1);
  VERIFY(m.rows()==rows+1 && m.cols()==cols+1);
  a2.matrix().resize(rows+1,cols+1);
  VERIFY(a2.rows()==rows+1 && a2.cols()==cols+1);
  v.array().resize(cols);
  VERIFY(v.size()==cols);
  a1.matrix().resize(cols);
  VERIFY(a1.size()==cols);
}

template<int>
void regression_bug_654()
{
  ArrayXf a = RowVectorXf(3);
  VectorXf v = Array<float,1,Dynamic>(3);
}

// Check propagation of LvalueBit through Array/Matrix-Wrapper
template<int>
void regrrssion_bug_1410()
{
  const Matrix4i M;
  const Array4i A;
  ArrayWrapper<const Matrix4i> MA = M.array();
  MA.row(0);
  MatrixWrapper<const Array4i> AM = A.matrix();
  AM.row(0);

  VERIFY((internal::traits<ArrayWrapper<const Matrix4i> >::Flags&LvalueBit)==0);
  VERIFY((internal::traits<MatrixWrapper<const Array4i> >::Flags&LvalueBit)==0);

  VERIFY((internal::traits<ArrayWrapper<Matrix4i> >::Flags&LvalueBit)==LvalueBit);
  VERIFY((internal::traits<MatrixWrapper<Array4i> >::Flags&LvalueBit)==LvalueBit);
}

void test_array_for_matrix()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array_for_matrix(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( array_for_matrix(Matrix2f()) );
    CALL_SUBTEST_3( array_for_matrix(Matrix4d()) );
    CALL_SUBTEST_4( array_for_matrix(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( array_for_matrix(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array_for_matrix(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( comparisons(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( comparisons(Matrix2f()) );
    CALL_SUBTEST_3( comparisons(Matrix4d()) );
    CALL_SUBTEST_5( comparisons(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( comparisons(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( cwise_min_max(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( cwise_min_max(Matrix2f()) );
    CALL_SUBTEST_3( cwise_min_max(Matrix4d()) );
    CALL_SUBTEST_5( cwise_min_max(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( cwise_min_max(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( lpNorm(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( lpNorm(Vector2f()) );
    CALL_SUBTEST_7( lpNorm(Vector3d()) );
    CALL_SUBTEST_8( lpNorm(Vector4f()) );
    CALL_SUBTEST_5( lpNorm(VectorXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_4( lpNorm(VectorXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  CALL_SUBTEST_5( lpNorm(VectorXf(0)) );
  CALL_SUBTEST_4( lpNorm(VectorXcf(0)) );
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_4( resize(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( resize(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( resize(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  CALL_SUBTEST_6( regression_bug_654<0>() );
  CALL_SUBTEST_6( regrrssion_bug_1410<0>() );
}
