// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 20013 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This unit test cannot be easily written to work with EIGEN_DEFAULT_TO_ROW_MAJOR
#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#undef EIGEN_DEFAULT_TO_ROW_MAJOR
#endif

#define TEST_ENABLE_TEMPORARY_TRACKING
#define TEST_CHECK_STATIC_ASSERTIONS
#include "main.h"

// test Ref.h

// Deal with i387 extended precision
#if EIGEN_ARCH_i386 && !(EIGEN_ARCH_x86_64)

#if EIGEN_COMP_GNUC_STRICT && EIGEN_GNUC_AT_LEAST(4,4)
#pragma GCC optimize ("-ffloat-store")
#else
#undef VERIFY_IS_EQUAL
#define VERIFY_IS_EQUAL(X,Y) VERIFY_IS_APPROX(X,Y)
#endif

#endif

template<typename MatrixType> void ref_matrix(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar,Dynamic,Dynamic,MatrixType::Options> DynMatrixType;
  typedef Matrix<RealScalar,Dynamic,Dynamic,MatrixType::Options> RealDynMatrixType;
  
  typedef Ref<MatrixType> RefMat;
  typedef Ref<DynMatrixType> RefDynMat;
  typedef Ref<const DynMatrixType> ConstRefDynMat;
  typedef Ref<RealDynMatrixType , 0, Stride<Dynamic,Dynamic> > RefRealMatWithStride;

  Index rows = m.rows(), cols = m.cols();
  
  MatrixType  m1 = MatrixType::Random(rows, cols),
              m2 = m1;
  
  Index i = internal::random<Index>(0,rows-1);
  Index j = internal::random<Index>(0,cols-1);
  Index brows = internal::random<Index>(1,rows-i);
  Index bcols = internal::random<Index>(1,cols-j);
  
  RefMat rm0 = m1;
  VERIFY_IS_EQUAL(rm0, m1);
  RefDynMat rm1 = m1;
  VERIFY_IS_EQUAL(rm1, m1);
  RefDynMat rm2 = m1.block(i,j,brows,bcols);
  VERIFY_IS_EQUAL(rm2, m1.block(i,j,brows,bcols));
  rm2.setOnes();
  m2.block(i,j,brows,bcols).setOnes();
  VERIFY_IS_EQUAL(m1, m2);
  
  m2.block(i,j,brows,bcols).setRandom();
  rm2 = m2.block(i,j,brows,bcols);
  VERIFY_IS_EQUAL(m1, m2);
  
  ConstRefDynMat rm3 = m1.block(i,j,brows,bcols);
  m1.block(i,j,brows,bcols) *= 2;
  m2.block(i,j,brows,bcols) *= 2;
  VERIFY_IS_EQUAL(rm3, m2.block(i,j,brows,bcols));
  RefRealMatWithStride rm4 = m1.real();
  VERIFY_IS_EQUAL(rm4, m2.real());
  rm4.array() += 1;
  m2.real().array() += 1;
  VERIFY_IS_EQUAL(m1, m2);
}

template<typename VectorType> void ref_vector(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;
  typedef typename VectorType::RealScalar RealScalar;
  typedef Matrix<Scalar,Dynamic,1,VectorType::Options> DynMatrixType;
  typedef Matrix<Scalar,Dynamic,Dynamic,ColMajor> MatrixType;
  typedef Matrix<RealScalar,Dynamic,1,VectorType::Options> RealDynMatrixType;
  
  typedef Ref<VectorType> RefMat;
  typedef Ref<DynMatrixType> RefDynMat;
  typedef Ref<const DynMatrixType> ConstRefDynMat;
  typedef Ref<RealDynMatrixType , 0, InnerStride<> > RefRealMatWithStride;
  typedef Ref<DynMatrixType , 0, InnerStride<> > RefMatWithStride;

  Index size = m.size();
  
  VectorType  v1 = VectorType::Random(size),
              v2 = v1;
  MatrixType mat1 = MatrixType::Random(size,size),
             mat2 = mat1,
             mat3 = MatrixType::Random(size,size);
  
  Index i = internal::random<Index>(0,size-1);
  Index bsize = internal::random<Index>(1,size-i);
  
  { RefMat    rm0 = v1;                   VERIFY_IS_EQUAL(rm0, v1); }
  { RefMat    rm0 = v1.block(0,0,size,1); VERIFY_IS_EQUAL(rm0, v1); }
  { RefDynMat rv1 = v1;                   VERIFY_IS_EQUAL(rv1, v1); }
  { RefDynMat rv1 = v1.block(0,0,size,1); VERIFY_IS_EQUAL(rv1, v1); }
  { VERIFY_RAISES_ASSERT( RefMat    rm0 = v1.block(0, 0, size, 0); EIGEN_UNUSED_VARIABLE(rm0); ); }
  if(VectorType::SizeAtCompileTime!=1)
  { VERIFY_RAISES_ASSERT( RefDynMat rv1 = v1.block(0, 0, size, 0); EIGEN_UNUSED_VARIABLE(rv1); ); }

  RefDynMat rv2 = v1.segment(i,bsize);
  VERIFY_IS_EQUAL(rv2, v1.segment(i,bsize));
  rv2.setOnes();
  v2.segment(i,bsize).setOnes();
  VERIFY_IS_EQUAL(v1, v2);
  
  v2.segment(i,bsize).setRandom();
  rv2 = v2.segment(i,bsize);
  VERIFY_IS_EQUAL(v1, v2);
  
  ConstRefDynMat rm3 = v1.segment(i,bsize);
  v1.segment(i,bsize) *= 2;
  v2.segment(i,bsize) *= 2;
  VERIFY_IS_EQUAL(rm3, v2.segment(i,bsize));
  
  RefRealMatWithStride rm4 = v1.real();
  VERIFY_IS_EQUAL(rm4, v2.real());
  rm4.array() += 1;
  v2.real().array() += 1;
  VERIFY_IS_EQUAL(v1, v2);
  
  RefMatWithStride rm5 = mat1.row(i).transpose();
  VERIFY_IS_EQUAL(rm5, mat1.row(i).transpose());
  rm5.array() += 1;
  mat2.row(i).array() += 1;
  VERIFY_IS_EQUAL(mat1, mat2);
  rm5.noalias() = rm4.transpose() * mat3;
  mat2.row(i) = v2.real().transpose() * mat3;
  VERIFY_IS_APPROX(mat1, mat2);
}

template<typename PlainObjectType> void check_const_correctness(const PlainObjectType&)
{
  // verify that ref-to-const don't have LvalueBit
  typedef typename internal::add_const<PlainObjectType>::type ConstPlainObjectType;
  VERIFY( !(internal::traits<Ref<ConstPlainObjectType> >::Flags & LvalueBit) );
  VERIFY( !(internal::traits<Ref<ConstPlainObjectType, Aligned> >::Flags & LvalueBit) );
  VERIFY( !(Ref<ConstPlainObjectType>::Flags & LvalueBit) );
  VERIFY( !(Ref<ConstPlainObjectType, Aligned>::Flags & LvalueBit) );
}

template<typename B>
EIGEN_DONT_INLINE void call_ref_1(Ref<VectorXf> a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_2(const Ref<const VectorXf>& a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_3(Ref<VectorXf,0,InnerStride<> > a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_4(const Ref<const VectorXf,0,InnerStride<> >& a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_5(Ref<MatrixXf,0,OuterStride<> > a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_6(const Ref<const MatrixXf,0,OuterStride<> >& a, const B &b) { VERIFY_IS_EQUAL(a,b); }
template<typename B>
EIGEN_DONT_INLINE void call_ref_7(Ref<Matrix<float,Dynamic,3> > a, const B &b) { VERIFY_IS_EQUAL(a,b); }

void call_ref()
{
  VectorXcf ca  = VectorXcf::Random(10);
  VectorXf a    = VectorXf::Random(10);
  RowVectorXf b = RowVectorXf::Random(10);
  MatrixXf A    = MatrixXf::Random(10,10);
  RowVector3f c = RowVector3f::Random();
  const VectorXf& ac(a);
  VectorBlock<VectorXf> ab(a,0,3);
  const VectorBlock<VectorXf> abc(a,0,3);
  

  VERIFY_EVALUATION_COUNT( call_ref_1(a,a), 0);
  VERIFY_EVALUATION_COUNT( call_ref_1(b,b.transpose()), 0);
//   call_ref_1(ac,a<c);           // does not compile because ac is const
  VERIFY_EVALUATION_COUNT( call_ref_1(ab,ab), 0);
  VERIFY_EVALUATION_COUNT( call_ref_1(a.head(4),a.head(4)), 0);
  VERIFY_EVALUATION_COUNT( call_ref_1(abc,abc), 0);
  VERIFY_EVALUATION_COUNT( call_ref_1(A.col(3),A.col(3)), 0);
//   call_ref_1(A.row(3),A.row(3));    // does not compile because innerstride!=1
  VERIFY_EVALUATION_COUNT( call_ref_3(A.row(3),A.row(3).transpose()), 0);
  VERIFY_EVALUATION_COUNT( call_ref_4(A.row(3),A.row(3).transpose()), 0);
//   call_ref_1(a+a, a+a);          // does not compile for obvious reason

  MatrixXf tmp = A*A.col(1);
  VERIFY_EVALUATION_COUNT( call_ref_2(A*A.col(1), tmp), 1);     // evaluated into a temp
  VERIFY_EVALUATION_COUNT( call_ref_2(ac.head(5),ac.head(5)), 0);
  VERIFY_EVALUATION_COUNT( call_ref_2(ac,ac), 0);
  VERIFY_EVALUATION_COUNT( call_ref_2(a,a), 0);
  VERIFY_EVALUATION_COUNT( call_ref_2(ab,ab), 0);
  VERIFY_EVALUATION_COUNT( call_ref_2(a.head(4),a.head(4)), 0);
  tmp = a+a;
  VERIFY_EVALUATION_COUNT( call_ref_2(a+a,tmp), 1);            // evaluated into a temp
  VERIFY_EVALUATION_COUNT( call_ref_2(ca.imag(),ca.imag()), 1);      // evaluated into a temp

  VERIFY_EVALUATION_COUNT( call_ref_4(ac.head(5),ac.head(5)), 0);
  tmp = a+a;
  VERIFY_EVALUATION_COUNT( call_ref_4(a+a,tmp), 1);           // evaluated into a temp
  VERIFY_EVALUATION_COUNT( call_ref_4(ca.imag(),ca.imag()), 0);

  VERIFY_EVALUATION_COUNT( call_ref_5(a,a), 0);
  VERIFY_EVALUATION_COUNT( call_ref_5(a.head(3),a.head(3)), 0);
  VERIFY_EVALUATION_COUNT( call_ref_5(A,A), 0);
//   call_ref_5(A.transpose(),A.transpose());   // does not compile because storage order does not match
  VERIFY_EVALUATION_COUNT( call_ref_5(A.block(1,1,2,2),A.block(1,1,2,2)), 0);
  VERIFY_EVALUATION_COUNT( call_ref_5(b,b), 0);             // storage order do not match, but this is a degenerate case that should work
  VERIFY_EVALUATION_COUNT( call_ref_5(a.row(3),a.row(3)), 0);

  VERIFY_EVALUATION_COUNT( call_ref_6(a,a), 0);
  VERIFY_EVALUATION_COUNT( call_ref_6(a.head(3),a.head(3)), 0);
  VERIFY_EVALUATION_COUNT( call_ref_6(A.row(3),A.row(3)), 1);           // evaluated into a temp thouth it could be avoided by viewing it as a 1xn matrix
  tmp = A+A;
  VERIFY_EVALUATION_COUNT( call_ref_6(A+A,tmp), 1);                // evaluated into a temp
  VERIFY_EVALUATION_COUNT( call_ref_6(A,A), 0);
  VERIFY_EVALUATION_COUNT( call_ref_6(A.transpose(),A.transpose()), 1);      // evaluated into a temp because the storage orders do not match
  VERIFY_EVALUATION_COUNT( call_ref_6(A.block(1,1,2,2),A.block(1,1,2,2)), 0);
  
  VERIFY_EVALUATION_COUNT( call_ref_7(c,c), 0);
}

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RowMatrixXd;
int test_ref_overload_fun1(Ref<MatrixXd> )       { return 1; }
int test_ref_overload_fun1(Ref<RowMatrixXd> )    { return 2; }
int test_ref_overload_fun1(Ref<MatrixXf> )       { return 3; }

int test_ref_overload_fun2(Ref<const MatrixXd> ) { return 4; }
int test_ref_overload_fun2(Ref<const MatrixXf> ) { return 5; }

void test_ref_ambiguous(const Ref<const ArrayXd> &A, Ref<ArrayXd> B)
{
  B = A;
  B = A - A;
}

// See also bug 969
void test_ref_overloads()
{
  MatrixXd Ad, Bd;
  RowMatrixXd rAd, rBd;
  VERIFY( test_ref_overload_fun1(Ad)==1 );
  VERIFY( test_ref_overload_fun1(rAd)==2 );
  
  MatrixXf Af, Bf;
  VERIFY( test_ref_overload_fun2(Ad)==4 );
  VERIFY( test_ref_overload_fun2(Ad+Bd)==4 );
  VERIFY( test_ref_overload_fun2(Af+Bf)==5 );
  
  ArrayXd A, B;
  test_ref_ambiguous(A, B);
}

void test_ref_fixed_size_assert()
{
  Vector4f v4;
  VectorXf vx(10);
  VERIFY_RAISES_STATIC_ASSERT( Ref<Vector3f> y = v4; (void)y; );
  VERIFY_RAISES_STATIC_ASSERT( Ref<Vector3f> y = vx.head<4>(); (void)y; );
  VERIFY_RAISES_STATIC_ASSERT( Ref<const Vector3f> y = v4; (void)y; );
  VERIFY_RAISES_STATIC_ASSERT( Ref<const Vector3f> y = vx.head<4>(); (void)y; );
  VERIFY_RAISES_STATIC_ASSERT( Ref<const Vector3f> y = 2*v4; (void)y; );
}

void test_ref()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( ref_vector(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( check_const_correctness(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( ref_vector(Vector4d()) );
    CALL_SUBTEST_2( check_const_correctness(Matrix4d()) );
    CALL_SUBTEST_3( ref_vector(Vector4cf()) );
    CALL_SUBTEST_4( ref_vector(VectorXcf(8)) );
    CALL_SUBTEST_5( ref_vector(VectorXi(12)) );
    CALL_SUBTEST_5( check_const_correctness(VectorXi(12)) );

    CALL_SUBTEST_1( ref_matrix(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( ref_matrix(Matrix4d()) );
    CALL_SUBTEST_1( ref_matrix(Matrix<float,3,5>()) );
    CALL_SUBTEST_4( ref_matrix(MatrixXcf(internal::random<int>(1,10),internal::random<int>(1,10))) );
    CALL_SUBTEST_4( ref_matrix(Matrix<std::complex<double>,10,15>()) );
    CALL_SUBTEST_5( ref_matrix(MatrixXi(internal::random<int>(1,10),internal::random<int>(1,10))) );
    CALL_SUBTEST_6( call_ref() );
  }
  
  CALL_SUBTEST_7( test_ref_overloads() );
  CALL_SUBTEST_7( test_ref_fixed_size_assert() );
}
