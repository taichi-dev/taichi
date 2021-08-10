// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename Scalar, int Size, int OtherSize> void symm(int size = Size, int othersize = OtherSize)
{
  typedef Matrix<Scalar, Size, Size> MatrixType;
  typedef Matrix<Scalar, Size, OtherSize> Rhs1;
  typedef Matrix<Scalar, OtherSize, Size> Rhs2;
  enum { order = OtherSize==1 ? 0 : RowMajor };
  typedef Matrix<Scalar, Size, OtherSize,order> Rhs3;

  Index rows = size;
  Index cols = size;

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols), m3;

  m1 = (m1+m1.adjoint()).eval();

  Rhs1 rhs1 = Rhs1::Random(cols, othersize), rhs12(cols, othersize), rhs13(cols, othersize);
  Rhs2 rhs2 = Rhs2::Random(othersize, rows), rhs22(othersize, rows), rhs23(othersize, rows);
  Rhs3 rhs3 = Rhs3::Random(cols, othersize), rhs32(cols, othersize), rhs33(cols, othersize);

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>();

  m2 = m1.template triangularView<Lower>();
  m3 = m2.template selfadjointView<Lower>();
  VERIFY_IS_EQUAL(m1, m3);
  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Lower>() * (s2*rhs1),
                   rhs13 = (s1*m1) * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).transpose().template selfadjointView<Upper>() * (s2*rhs1),
                   rhs13 = (s1*m1.transpose()) * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Lower>().transpose() * (s2*rhs1),
                   rhs13 = (s1*m1.transpose()) * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).conjugate().template selfadjointView<Lower>() * (s2*rhs1),
                   rhs13 = (s1*m1).conjugate() * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Lower>().conjugate() * (s2*rhs1),
                   rhs13 = (s1*m1).conjugate() * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).adjoint().template selfadjointView<Upper>() * (s2*rhs1),
                   rhs13 = (s1*m1).adjoint() * (s2*rhs1));

  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Lower>().adjoint() * (s2*rhs1),
                   rhs13 = (s1*m1).adjoint() * (s2*rhs1));

  m2 = m1.template triangularView<Upper>(); rhs12.setRandom(); rhs13 = rhs12;
  m3 = m2.template selfadjointView<Upper>();
  VERIFY_IS_EQUAL(m1, m3);
  VERIFY_IS_APPROX(rhs12 += (s1*m2).template selfadjointView<Upper>() * (s2*rhs1),
                   rhs13 += (s1*m1) * (s2*rhs1));

  m2 = m1.template triangularView<Lower>();
  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Lower>() * (s2*rhs2.adjoint()),
                   rhs13 = (s1*m1) * (s2*rhs2.adjoint()));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<Upper>() * (s2*rhs2.adjoint()),
                   rhs13 = (s1*m1) * (s2*rhs2.adjoint()));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs12 = (s1*m2.adjoint()).template selfadjointView<Lower>() * (s2*rhs2.adjoint()),
                   rhs13 = (s1*m1.adjoint()) * (s2*rhs2.adjoint()));

  // test row major = <...>
  m2 = m1.template triangularView<Lower>(); rhs32.setRandom(); rhs13 = rhs32;
  VERIFY_IS_APPROX(rhs32.noalias() -= (s1*m2).template selfadjointView<Lower>() * (s2*rhs3),
                   rhs13 -= (s1*m1) * (s2 * rhs3));

  m2 = m1.template triangularView<Upper>();
  VERIFY_IS_APPROX(rhs32.noalias() = (s1*m2.adjoint()).template selfadjointView<Lower>() * (s2*rhs3).conjugate(),
                   rhs13 = (s1*m1.adjoint()) * (s2*rhs3).conjugate());


  m2 = m1.template triangularView<Upper>(); rhs13 = rhs12;
  VERIFY_IS_APPROX(rhs12.noalias() += s1 * ((m2.adjoint()).template selfadjointView<Lower>() * (s2*rhs3).conjugate()),
                   rhs13 += (s1*m1.adjoint()) * (s2*rhs3).conjugate());

  m2 = m1.template triangularView<Lower>();
  VERIFY_IS_APPROX(rhs22 = (rhs2) * (m2).template selfadjointView<Lower>(), rhs23 = (rhs2) * (m1));
  VERIFY_IS_APPROX(rhs22 = (s2*rhs2) * (s1*m2).template selfadjointView<Lower>(), rhs23 = (s2*rhs2) * (s1*m1));

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*cols,2*othersize);
    Map<Rhs1,0,Stride<Dynamic,2> > map1(buffer.data(),cols,othersize,Stride<Dynamic,2>(2*rows,2));
    buffer.setZero();
    VERIFY_IS_APPROX( map1.noalias()  = (s1*m2).template selfadjointView<Lower>() * (s2*rhs1),
                      rhs13 = (s1*m1) * (s2*rhs1));

    Map<Rhs2,0,Stride<Dynamic,2> > map2(buffer.data(),rhs22.rows(),rhs22.cols(),Stride<Dynamic,2>(2*rhs22.outerStride(),2));
    buffer.setZero();
    VERIFY_IS_APPROX(map2 = (rhs2) * (m2).template selfadjointView<Lower>(), rhs23 = (rhs2) * (m1));
  }
}

void test_product_symm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    CALL_SUBTEST_1(( symm<float,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
    CALL_SUBTEST_2(( symm<double,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
    CALL_SUBTEST_3(( symm<std::complex<float>,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2),internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2)) ));
    CALL_SUBTEST_4(( symm<std::complex<double>,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2),internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2)) ));

    CALL_SUBTEST_5(( symm<float,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
    CALL_SUBTEST_6(( symm<double,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
    CALL_SUBTEST_7(( symm<std::complex<float>,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
    CALL_SUBTEST_8(( symm<std::complex<double>,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE)) ));
  }
}
