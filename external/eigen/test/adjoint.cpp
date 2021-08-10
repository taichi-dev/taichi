// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT

#include "main.h"

template<bool IsInteger> struct adjoint_specific;

template<> struct adjoint_specific<true> {
  template<typename Vec, typename Mat, typename Scalar>
  static void run(const Vec& v1, const Vec& v2, Vec& v3, const Mat& square, Scalar s1, Scalar s2) {
    VERIFY(test_isApproxWithRef((s1 * v1 + s2 * v2).dot(v3),     numext::conj(s1) * v1.dot(v3) + numext::conj(s2) * v2.dot(v3), 0));
    VERIFY(test_isApproxWithRef(v3.dot(s1 * v1 + s2 * v2),       s1*v3.dot(v1)+s2*v3.dot(v2), 0));
    
    // check compatibility of dot and adjoint
    VERIFY(test_isApproxWithRef(v1.dot(square * v2), (square.adjoint() * v1).dot(v2), 0));
  }
};

template<> struct adjoint_specific<false> {
  template<typename Vec, typename Mat, typename Scalar>
  static void run(const Vec& v1, const Vec& v2, Vec& v3, const Mat& square, Scalar s1, Scalar s2) {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    using std::abs;
    
    RealScalar ref = NumTraits<Scalar>::IsInteger ? RealScalar(0) : (std::max)((s1 * v1 + s2 * v2).norm(),v3.norm());
    VERIFY(test_isApproxWithRef((s1 * v1 + s2 * v2).dot(v3),     numext::conj(s1) * v1.dot(v3) + numext::conj(s2) * v2.dot(v3), ref));
    VERIFY(test_isApproxWithRef(v3.dot(s1 * v1 + s2 * v2),       s1*v3.dot(v1)+s2*v3.dot(v2), ref));
  
    VERIFY_IS_APPROX(v1.squaredNorm(),                v1.norm() * v1.norm());
    // check normalized() and normalize()
    VERIFY_IS_APPROX(v1, v1.norm() * v1.normalized());
    v3 = v1;
    v3.normalize();
    VERIFY_IS_APPROX(v1, v1.norm() * v3);
    VERIFY_IS_APPROX(v3, v1.normalized());
    VERIFY_IS_APPROX(v3.norm(), RealScalar(1));

    // check null inputs
    VERIFY_IS_APPROX((v1*0).normalized(), (v1*0));
#if (!EIGEN_ARCH_i386) || defined(EIGEN_VECTORIZE)
    RealScalar very_small = (std::numeric_limits<RealScalar>::min)();
    VERIFY( (v1*very_small).norm() == 0 );
    VERIFY_IS_APPROX((v1*very_small).normalized(), (v1*very_small));
    v3 = v1*very_small;
    v3.normalize();
    VERIFY_IS_APPROX(v3, (v1*very_small));
#endif
    
    // check compatibility of dot and adjoint
    ref = NumTraits<Scalar>::IsInteger ? 0 : (std::max)((std::max)(v1.norm(),v2.norm()),(std::max)((square * v2).norm(),(square.adjoint() * v1).norm()));
    VERIFY(internal::isMuchSmallerThan(abs(v1.dot(square * v2) - (square.adjoint() * v1).dot(v2)), ref, test_precision<Scalar>()));
    
    // check that Random().normalized() works: tricky as the random xpr must be evaluated by
    // normalized() in order to produce a consistent result.
    VERIFY_IS_APPROX(Vec::Random(v1.size()).normalized().norm(), RealScalar(1));
  }
};

template<typename MatrixType> void adjoint(const MatrixType& m)
{
  /* this test covers the following files:
     Transpose.h Conjugate.h Dot.h
  */
  using std::abs;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  const Index PacketSize = internal::packet_traits<Scalar>::size;
  
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             square = SquareMatrixType::Random(rows, rows);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows),
             v3 = VectorType::Random(rows),
             vzero = VectorType::Zero(rows);

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>();

  // check basic compatibility of adjoint, transpose, conjugate
  VERIFY_IS_APPROX(m1.transpose().conjugate().adjoint(),    m1);
  VERIFY_IS_APPROX(m1.adjoint().conjugate().transpose(),    m1);

  // check multiplicative behavior
  VERIFY_IS_APPROX((m1.adjoint() * m2).adjoint(),           m2.adjoint() * m1);
  VERIFY_IS_APPROX((s1 * m1).adjoint(),                     numext::conj(s1) * m1.adjoint());

  // check basic properties of dot, squaredNorm
  VERIFY_IS_APPROX(numext::conj(v1.dot(v2)),               v2.dot(v1));
  VERIFY_IS_APPROX(numext::real(v1.dot(v1)),               v1.squaredNorm());
  
  adjoint_specific<NumTraits<Scalar>::IsInteger>::run(v1, v2, v3, square, s1, s2);
  
  VERIFY_IS_MUCH_SMALLER_THAN(abs(vzero.dot(v1)),  static_cast<RealScalar>(1));
  
  // like in testBasicStuff, test operator() to check const-qualification
  Index r = internal::random<Index>(0, rows-1),
      c = internal::random<Index>(0, cols-1);
  VERIFY_IS_APPROX(m1.conjugate()(r,c), numext::conj(m1(r,c)));
  VERIFY_IS_APPROX(m1.adjoint()(c,r), numext::conj(m1(r,c)));

  // check inplace transpose
  m3 = m1;
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3,m1.transpose());
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3,m1);
  
  if(PacketSize<m3.rows() && PacketSize<m3.cols())
  {
    m3 = m1;
    Index i = internal::random<Index>(0,m3.rows()-PacketSize);
    Index j = internal::random<Index>(0,m3.cols()-PacketSize);
    m3.template block<PacketSize,PacketSize>(i,j).transposeInPlace();
    VERIFY_IS_APPROX( (m3.template block<PacketSize,PacketSize>(i,j)), (m1.template block<PacketSize,PacketSize>(i,j).transpose()) );
    m3.template block<PacketSize,PacketSize>(i,j).transposeInPlace();
    VERIFY_IS_APPROX(m3,m1);
  }

  // check inplace adjoint
  m3 = m1;
  m3.adjointInPlace();
  VERIFY_IS_APPROX(m3,m1.adjoint());
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3,m1.conjugate());

  // check mixed dot product
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, 1> RealVectorType;
  RealVectorType rv1 = RealVectorType::Random(rows);
  VERIFY_IS_APPROX(v1.dot(rv1.template cast<Scalar>()), v1.dot(rv1));
  VERIFY_IS_APPROX(rv1.template cast<Scalar>().dot(v1), rv1.dot(v1));
}

void test_adjoint()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( adjoint(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( adjoint(Matrix3d()) );
    CALL_SUBTEST_3( adjoint(Matrix4f()) );
    
    CALL_SUBTEST_4( adjoint(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_5( adjoint(MatrixXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( adjoint(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    
    // Complement for 128 bits vectorization:
    CALL_SUBTEST_8( adjoint(Matrix2d()) );
    CALL_SUBTEST_9( adjoint(Matrix<int,4,4>()) );
    
    // 256 bits vectorization:
    CALL_SUBTEST_10( adjoint(Matrix<float,8,8>()) );
    CALL_SUBTEST_11( adjoint(Matrix<double,4,4>()) );
    CALL_SUBTEST_12( adjoint(Matrix<int,8,8>()) );
  }
  // test a large static matrix only once
  CALL_SUBTEST_7( adjoint(Matrix<float, 100, 100>()) );

#ifdef EIGEN_TEST_PART_13
  {
    MatrixXcf a(10,10), b(10,10);
    VERIFY_RAISES_ASSERT(a = a.transpose());
    VERIFY_RAISES_ASSERT(a = a.transpose() + b);
    VERIFY_RAISES_ASSERT(a = b + a.transpose());
    VERIFY_RAISES_ASSERT(a = a.conjugate().transpose());
    VERIFY_RAISES_ASSERT(a = a.adjoint());
    VERIFY_RAISES_ASSERT(a = a.adjoint() + b);
    VERIFY_RAISES_ASSERT(a = b + a.adjoint());

    // no assertion should be triggered for these cases:
    a.transpose() = a.transpose();
    a.transpose() += a.transpose();
    a.transpose() += a.transpose() + b;
    a.transpose() = a.adjoint();
    a.transpose() += a.adjoint();
    a.transpose() += a.adjoint() + b;

    // regression tests for check_for_aliasing
    MatrixXd c(10,10);
    c = 1.0 * MatrixXd::Ones(10,10) + c;
    c = MatrixXd::Ones(10,10) * 1.0 + c;
    c = c + MatrixXd::Ones(10,10) .cwiseProduct( MatrixXd::Zero(10,10) );
    c = MatrixXd::Ones(10,10) * MatrixXd::Zero(10,10);
  }
#endif
}

