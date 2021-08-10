// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void product_extra(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic,
                         MatrixType::Flags&RowMajorBit> OtherMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows),
             res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols),
             res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows), vrres(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>(),
         s3 = internal::random<Scalar>();

  VERIFY_IS_APPROX(m3.noalias() = m1 * m2.adjoint(),                 m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * square.adjoint(),   m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * m2,                 m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (s1 * m1.adjoint()) * m2,          (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = ((s1 * m1).adjoint()) * m2,        (numext::conj(s1) * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (- m1.adjoint() * s1) * (s3 * m2), (- m1.adjoint()  * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX(m3.noalias() = (s2 * m1.adjoint() * s1) * m2,     (s2 * m1.adjoint()  * s1).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1*s2) * s1*m2.adjoint(),        (-m1*s2).eval() * (s1*m2.adjoint()).eval());

  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX( m1.adjoint() * (s1*m2).conjugate(), (m1.adjoint()).eval() * ((s1*m2).conjugate()).eval());


  // test all possible conjugate combinations for the four matrix-vector product cases:

  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()),
                   (-m1*s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2.conjugate()).eval());

  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.transpose()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.adjoint() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.adjoint()*s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.transpose()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.transpose()).eval());
  VERIFY_IS_APPROX((-m1.transpose() * s2) * (s1 * v1.adjoint()),
                   (-m1.transpose()*s2).eval() * (s1 * v1.adjoint()).eval());
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());

  VERIFY_IS_APPROX((s1 * v1) * (-m1.conjugate() * s2),
                   (s1 * v1).eval() * (-m1.conjugate()*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1 * s2),
                   (s1 * v1.conjugate()).eval() * (-m1*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1.conjugate() * s2),
                   (s1 * v1.conjugate()).eval() * (-m1.conjugate()*s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());

  // test the vector-matrix product with non aligned starts
  Index i = internal::random<Index>(0,m1.rows()-2);
  Index j = internal::random<Index>(0,m1.cols()-2);
  Index r = internal::random<Index>(1,m1.rows()-i);
  Index c = internal::random<Index>(1,m1.cols()-j);
  Index i2 = internal::random<Index>(0,m1.rows()-1);
  Index j2 = internal::random<Index>(0,m1.cols()-1);

  VERIFY_IS_APPROX(m1.col(j2).adjoint() * m1.block(0,j,m1.rows(),c), m1.col(j2).adjoint().eval() * m1.block(0,j,m1.rows(),c).eval());
  VERIFY_IS_APPROX(m1.block(i,0,r,m1.cols()) * m1.row(i2).adjoint(), m1.block(i,0,r,m1.cols()).eval() * m1.row(i2).adjoint().eval());
  
  // regression test
  MatrixType tmp = m1 * m1.adjoint() * s1;
  VERIFY_IS_APPROX(tmp, m1 * m1.adjoint() * s1);

  // regression test for bug 1343, assignment to arrays
  Array<Scalar,Dynamic,1> a1 = m1 * vc2;
  VERIFY_IS_APPROX(a1.matrix(),m1*vc2);
  Array<Scalar,Dynamic,1> a2 = s1 * (m1 * vc2);
  VERIFY_IS_APPROX(a2.matrix(),s1*m1*vc2);
  Array<Scalar,1,Dynamic> a3 = v1 * m1;
  VERIFY_IS_APPROX(a3.matrix(),v1*m1);
  Array<Scalar,Dynamic,Dynamic> a4 = m1 * m2.adjoint();
  VERIFY_IS_APPROX(a4.matrix(),m1*m2.adjoint());
}

// Regression test for bug reported at http://forum.kde.org/viewtopic.php?f=74&t=96947
void mat_mat_scalar_scalar_product()
{
  Eigen::Matrix2Xd dNdxy(2, 3);
  dNdxy << -0.5, 0.5, 0,
           -0.3, 0, 0.3;
  double det = 6.0, wt = 0.5;
  VERIFY_IS_APPROX(dNdxy.transpose()*dNdxy*det*wt, det*wt*dNdxy.transpose()*dNdxy);
}

template <typename MatrixType> 
void zero_sized_objects(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  const int PacketSize  = internal::packet_traits<Scalar>::size;
  const int PacketSize1 = PacketSize>1 ?  PacketSize-1 : 1;
  Index rows = m.rows();
  Index cols = m.cols();
  
  {
    MatrixType res, a(rows,0), b(0,cols);
    VERIFY_IS_APPROX( (res=a*b), MatrixType::Zero(rows,cols) );
    VERIFY_IS_APPROX( (res=a*a.transpose()), MatrixType::Zero(rows,rows) );
    VERIFY_IS_APPROX( (res=b.transpose()*b), MatrixType::Zero(cols,cols) );
    VERIFY_IS_APPROX( (res=b.transpose()*a.transpose()), MatrixType::Zero(cols,rows) );
  }
  
  {
    MatrixType res, a(rows,cols), b(cols,0);
    res = a*b;
    VERIFY(res.rows()==rows && res.cols()==0);
    b.resize(0,rows);
    res = b*a;
    VERIFY(res.rows()==0 && res.cols()==cols);
  }
  
  {
    Matrix<Scalar,PacketSize,0> a;
    Matrix<Scalar,0,1> b;
    Matrix<Scalar,PacketSize,1> res;
    VERIFY_IS_APPROX( (res=a*b), MatrixType::Zero(PacketSize,1) );
    VERIFY_IS_APPROX( (res=a.lazyProduct(b)), MatrixType::Zero(PacketSize,1) );
  }
  
  {
    Matrix<Scalar,PacketSize1,0> a;
    Matrix<Scalar,0,1> b;
    Matrix<Scalar,PacketSize1,1> res;
    VERIFY_IS_APPROX( (res=a*b), MatrixType::Zero(PacketSize1,1) );
    VERIFY_IS_APPROX( (res=a.lazyProduct(b)), MatrixType::Zero(PacketSize1,1) );
  }
  
  {
    Matrix<Scalar,PacketSize,Dynamic> a(PacketSize,0);
    Matrix<Scalar,Dynamic,1> b(0,1);
    Matrix<Scalar,PacketSize,1> res;
    VERIFY_IS_APPROX( (res=a*b), MatrixType::Zero(PacketSize,1) );
    VERIFY_IS_APPROX( (res=a.lazyProduct(b)), MatrixType::Zero(PacketSize,1) );
  }
  
  {
    Matrix<Scalar,PacketSize1,Dynamic> a(PacketSize1,0);
    Matrix<Scalar,Dynamic,1> b(0,1);
    Matrix<Scalar,PacketSize1,1> res;
    VERIFY_IS_APPROX( (res=a*b), MatrixType::Zero(PacketSize1,1) );
    VERIFY_IS_APPROX( (res=a.lazyProduct(b)), MatrixType::Zero(PacketSize1,1) );
  }
}

template<int>
void bug_127()
{
  // Bug 127
  //
  // a product of the form lhs*rhs with
  //
  // lhs:
  // rows = 1, cols = 4
  // RowsAtCompileTime = 1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 1, MaxColsAtCompileTime = 5
  //
  // rhs:
  // rows = 4, cols = 0
  // RowsAtCompileTime = -1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 5, MaxColsAtCompileTime = 1
  //
  // was failing on a runtime assertion, because it had been mis-compiled as a dot product because Product.h was using the
  // max-sizes to detect size 1 indicating vectors, and that didn't account for 0-sized object with max-size 1.

  Matrix<float,1,Dynamic,RowMajor,1,5> a(1,4);
  Matrix<float,Dynamic,Dynamic,ColMajor,5,1> b(4,0);
  a*b;
}

template<int> void bug_817()
{
  ArrayXXf B = ArrayXXf::Random(10,10), C;
  VectorXf x = VectorXf::Random(10);
  C = (x.transpose()*B.matrix());
  B = (x.transpose()*B.matrix());
  VERIFY_IS_APPROX(B,C);
}

template<int>
void unaligned_objects()
{
  // Regression test for the bug reported here:
  // http://forum.kde.org/viewtopic.php?f=74&t=107541
  // Recall the matrix*vector kernel avoid unaligned loads by loading two packets and then reassemble then.
  // There was a mistake in the computation of the valid range for fully unaligned objects: in some rare cases,
  // memory was read outside the allocated matrix memory. Though the values were not used, this might raise segfault.
  for(int m=450;m<460;++m)
  {
    for(int n=8;n<12;++n)
    {
      MatrixXf M(m, n);
      VectorXf v1(n), r1(500);
      RowVectorXf v2(m), r2(16);

      M.setRandom();
      v1.setRandom();
      v2.setRandom();
      for(int o=0; o<4; ++o)
      {
        r1.segment(o,m).noalias() = M * v1;
        VERIFY_IS_APPROX(r1.segment(o,m), M * MatrixXf(v1));
        r2.segment(o,n).noalias() = v2 * M;
        VERIFY_IS_APPROX(r2.segment(o,n), MatrixXf(v2) * M);
      }
    }
  }
}

template<typename T>
EIGEN_DONT_INLINE
Index test_compute_block_size(Index m, Index n, Index k)
{
  Index mc(m), nc(n), kc(k);
  internal::computeProductBlockingSizes<T,T>(kc, mc, nc);
  return kc+mc+nc;
}

template<typename T>
Index compute_block_size()
{
  Index ret = 0;
  ret += test_compute_block_size<T>(0,1,1);
  ret += test_compute_block_size<T>(1,0,1);
  ret += test_compute_block_size<T>(1,1,0);
  ret += test_compute_block_size<T>(0,0,1);
  ret += test_compute_block_size<T>(0,1,0);
  ret += test_compute_block_size<T>(1,0,0);
  ret += test_compute_block_size<T>(0,0,0);
  return ret;
}

template<typename>
void aliasing_with_resize()
{
  Index m = internal::random<Index>(10,50);
  Index n = internal::random<Index>(10,50);
  MatrixXd A, B, C(m,n), D(m,m);
  VectorXd a, b, c(n);
  C.setRandom();
  D.setRandom();
  c.setRandom();
  double s = internal::random<double>(1,10);

  A = C;
  B = A * A.transpose();
  A = A * A.transpose();
  VERIFY_IS_APPROX(A,B);

  A = C;
  B = (A * A.transpose())/s;
  A = (A * A.transpose())/s;
  VERIFY_IS_APPROX(A,B);

  A = C;
  B = (A * A.transpose()) + D;
  A = (A * A.transpose()) + D;
  VERIFY_IS_APPROX(A,B);

  A = C;
  B = D + (A * A.transpose());
  A = D + (A * A.transpose());
  VERIFY_IS_APPROX(A,B);

  A = C;
  B = s * (A * A.transpose());
  A = s * (A * A.transpose());
  VERIFY_IS_APPROX(A,B);

  A = C;
  a = c;
  b = (A * a)/s;
  a = (A * a)/s;
  VERIFY_IS_APPROX(a,b);
}

template<int>
void bug_1308()
{
  int n = 10;
  MatrixXd r(n,n);
  VectorXd v = VectorXd::Random(n);
  r = v * RowVectorXd::Ones(n);
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n));
  r = VectorXd::Ones(n) * v.transpose();
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n).transpose());

  Matrix4d ones44 = Matrix4d::Ones();
  Matrix4d m44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(m44,Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias()=ones44*Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias()=ones44.transpose()*Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias()=Matrix4d::Ones()*ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias()=Matrix4d::Ones()*ones44.transpose(), Matrix4d::Constant(4));

  typedef Matrix<double,4,4,RowMajor> RMatrix4d;
  RMatrix4d r44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(r44,Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=ones44*Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=ones44.transpose()*Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=Matrix4d::Ones()*ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=Matrix4d::Ones()*ones44.transpose(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=ones44*RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=ones44.transpose()*RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=RMatrix4d::Ones()*ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias()=RMatrix4d::Ones()*ones44.transpose(), Matrix4d::Constant(4));

//   RowVector4d r4;
  m44.setOnes();
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.row(0).transpose() * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.col(0) * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.row(0), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.col(0).transpose(), ones44);
}

void test_product_extra()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( product_extra(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( product_extra(MatrixXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_2( mat_mat_scalar_scalar_product() );
    CALL_SUBTEST_3( product_extra(MatrixXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_4( product_extra(MatrixXcd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2), internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))) );
    CALL_SUBTEST_1( zero_sized_objects(MatrixXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  CALL_SUBTEST_5( bug_127<0>() );
  CALL_SUBTEST_5( bug_817<0>() );
  CALL_SUBTEST_5( bug_1308<0>() );
  CALL_SUBTEST_6( unaligned_objects<0>() );
  CALL_SUBTEST_7( compute_block_size<float>() );
  CALL_SUBTEST_7( compute_block_size<double>() );
  CALL_SUBTEST_7( compute_block_size<std::complex<double> >() );
  CALL_SUBTEST_8( aliasing_with_resize<void>() );

}
