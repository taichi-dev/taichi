// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

static bool g_called;
#define EIGEN_SCALAR_BINARY_OP_PLUGIN { g_called |= (!internal::is_same<LhsScalar,RhsScalar>::value); }

#include "main.h"

template<typename MatrixType> void linearStructure(const MatrixType& m)
{
  using std::abs;
  /* this test covers the following files:
     CwiseUnaryOp.h, CwiseBinaryOp.h, SelfCwiseBinaryOp.h 
  */
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  Scalar s1 = internal::random<Scalar>();
  while (abs(s1)<RealScalar(1e-3)) s1 = internal::random<Scalar>();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  VERIFY_IS_APPROX(-(-m1),                  m1);
  VERIFY_IS_APPROX(m1+m1,                   2*m1);
  VERIFY_IS_APPROX(m1+m2-m1,                m2);
  VERIFY_IS_APPROX(-m2+m1+m2,               m1);
  VERIFY_IS_APPROX(m1*s1,                   s1*m1);
  VERIFY_IS_APPROX((m1+m2)*s1,              s1*m1+s1*m2);
  VERIFY_IS_APPROX((-m1+m2)*s1,             -s1*m1+s1*m2);
  m3 = m2; m3 += m1;
  VERIFY_IS_APPROX(m3,                      m1+m2);
  m3 = m2; m3 -= m1;
  VERIFY_IS_APPROX(m3,                      m2-m1);
  m3 = m2; m3 *= s1;
  VERIFY_IS_APPROX(m3,                      s1*m2);
  if(!NumTraits<Scalar>::IsInteger)
  {
    m3 = m2; m3 /= s1;
    VERIFY_IS_APPROX(m3,                    m2/s1);
  }

  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX((-m1)(r,c), -(m1(r,c)));
  VERIFY_IS_APPROX((m1-m2)(r,c), (m1(r,c))-(m2(r,c)));
  VERIFY_IS_APPROX((m1+m2)(r,c), (m1(r,c))+(m2(r,c)));
  VERIFY_IS_APPROX((s1*m1)(r,c), s1*(m1(r,c)));
  VERIFY_IS_APPROX((m1*s1)(r,c), (m1(r,c))*s1);
  if(!NumTraits<Scalar>::IsInteger)
    VERIFY_IS_APPROX((m1/s1)(r,c), (m1(r,c))/s1);

  // use .block to disable vectorization and compare to the vectorized version
  VERIFY_IS_APPROX(m1+m1.block(0,0,rows,cols), m1+m1);
  VERIFY_IS_APPROX(m1.cwiseProduct(m1.block(0,0,rows,cols)), m1.cwiseProduct(m1));
  VERIFY_IS_APPROX(m1 - m1.block(0,0,rows,cols), m1 - m1);
  VERIFY_IS_APPROX(m1.block(0,0,rows,cols) * s1, m1 * s1);
}

// Make sure that complex * real and real * complex are properly optimized
template<typename MatrixType> void real_complex(DenseIndex rows = MatrixType::RowsAtCompileTime, DenseIndex cols = MatrixType::ColsAtCompileTime)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  
  RealScalar s = internal::random<RealScalar>();
  MatrixType m1 = MatrixType::Random(rows, cols);
  
  g_called = false;
  VERIFY_IS_APPROX(s*m1, Scalar(s)*m1);
  VERIFY(g_called && "real * matrix<complex> not properly optimized");
  
  g_called = false;
  VERIFY_IS_APPROX(m1*s, m1*Scalar(s));
  VERIFY(g_called && "matrix<complex> * real not properly optimized");
  
  g_called = false;
  VERIFY_IS_APPROX(m1/s, m1/Scalar(s));
  VERIFY(g_called && "matrix<complex> / real not properly optimized");

  g_called = false;
  VERIFY_IS_APPROX(s+m1.array(), Scalar(s)+m1.array());
  VERIFY(g_called && "real + matrix<complex> not properly optimized");

  g_called = false;
  VERIFY_IS_APPROX(m1.array()+s, m1.array()+Scalar(s));
  VERIFY(g_called && "matrix<complex> + real not properly optimized");

  g_called = false;
  VERIFY_IS_APPROX(s-m1.array(), Scalar(s)-m1.array());
  VERIFY(g_called && "real - matrix<complex> not properly optimized");

  g_called = false;
  VERIFY_IS_APPROX(m1.array()-s, m1.array()-Scalar(s));
  VERIFY(g_called && "matrix<complex> - real not properly optimized");
}

void test_linearstructure()
{
  g_called = true;
  VERIFY(g_called); // avoid `unneeded-internal-declaration` warning.
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( linearStructure(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( linearStructure(Matrix2f()) );
    CALL_SUBTEST_3( linearStructure(Vector3d()) );
    CALL_SUBTEST_4( linearStructure(Matrix4d()) );
    CALL_SUBTEST_5( linearStructure(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_6( linearStructure(MatrixXf (internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_7( linearStructure(MatrixXi (internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_8( linearStructure(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_9( linearStructure(ArrayXXf (internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_10( linearStructure(ArrayXXcf (internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    
    CALL_SUBTEST_11( real_complex<Matrix4cd>() );
    CALL_SUBTEST_11( real_complex<MatrixXcf>(10,10) );
    CALL_SUBTEST_11( real_complex<ArrayXXcf>(10,10) );
  }
  
#ifdef EIGEN_TEST_PART_4
  {
    // make sure that /=scalar and /scalar do not overflow
    // rational: 1.0/4.94e-320 overflow, but m/4.94e-320 should not
    Matrix4d m2, m3;
    m3 = m2 =  Matrix4d::Random()*1e-20;
    m2 = m2 / 4.9e-320;
    VERIFY_IS_APPROX(m2.cwiseQuotient(m2), Matrix4d::Ones());
    m3 /= 4.9e-320;
    VERIFY_IS_APPROX(m3.cwiseQuotient(m3), Matrix4d::Ones());
    
    
  }
#endif
}
