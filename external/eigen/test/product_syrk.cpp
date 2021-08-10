// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void syrk(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime, RowMajor> RMatrixType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic> Rhs1;
  typedef Matrix<Scalar, Dynamic, MatrixType::RowsAtCompileTime> Rhs2;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic,RowMajor> Rhs3;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3 = MatrixType::Random(rows, cols);
  RMatrixType rm2 = MatrixType::Random(rows, cols);

  Rhs1 rhs1 = Rhs1::Random(internal::random<int>(1,320), cols); Rhs1 rhs11 = Rhs1::Random(rhs1.rows(), cols);
  Rhs2 rhs2 = Rhs2::Random(rows, internal::random<int>(1,320)); Rhs2 rhs22 = Rhs2::Random(rows, rhs2.cols());
  Rhs3 rhs3 = Rhs3::Random(internal::random<int>(1,320), rows);

  Scalar s1 = internal::random<Scalar>();
  
  Index c = internal::random<Index>(0,cols-1);

  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Lower>().rankUpdate(rhs2,s1)._expression()),
                   ((s1 * rhs2 * rhs2.adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  m2.setZero();
  VERIFY_IS_APPROX(((m2.template triangularView<Lower>() += s1 * rhs2  * rhs22.adjoint()).nestedExpression()),
                   ((s1 * rhs2 * rhs22.adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));

  
  m2.setZero();
  VERIFY_IS_APPROX(m2.template selfadjointView<Upper>().rankUpdate(rhs2,s1)._expression(),
                   (s1 * rhs2 * rhs2.adjoint()).eval().template triangularView<Upper>().toDenseMatrix());
  m2.setZero();
  VERIFY_IS_APPROX((m2.template triangularView<Upper>() += s1 * rhs22 * rhs2.adjoint()).nestedExpression(),
                   (s1 * rhs22 * rhs2.adjoint()).eval().template triangularView<Upper>().toDenseMatrix());

  
  m2.setZero();
  VERIFY_IS_APPROX(m2.template selfadjointView<Lower>().rankUpdate(rhs1.adjoint(),s1)._expression(),
                   (s1 * rhs1.adjoint() * rhs1).eval().template triangularView<Lower>().toDenseMatrix());
  m2.setZero();
  VERIFY_IS_APPROX((m2.template triangularView<Lower>() += s1 * rhs11.adjoint() * rhs1).nestedExpression(),
                   (s1 * rhs11.adjoint() * rhs1).eval().template triangularView<Lower>().toDenseMatrix());
  
  
  m2.setZero();
  VERIFY_IS_APPROX(m2.template selfadjointView<Upper>().rankUpdate(rhs1.adjoint(),s1)._expression(),
                   (s1 * rhs1.adjoint() * rhs1).eval().template triangularView<Upper>().toDenseMatrix());
  VERIFY_IS_APPROX((m2.template triangularView<Upper>() = s1 * rhs1.adjoint() * rhs11).nestedExpression(),
                   (s1 * rhs1.adjoint() * rhs11).eval().template triangularView<Upper>().toDenseMatrix());

  
  m2.setZero();
  VERIFY_IS_APPROX(m2.template selfadjointView<Lower>().rankUpdate(rhs3.adjoint(),s1)._expression(),
                   (s1 * rhs3.adjoint() * rhs3).eval().template triangularView<Lower>().toDenseMatrix());

  m2.setZero();
  VERIFY_IS_APPROX(m2.template selfadjointView<Upper>().rankUpdate(rhs3.adjoint(),s1)._expression(),
                   (s1 * rhs3.adjoint() * rhs3).eval().template triangularView<Upper>().toDenseMatrix());
                   
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Lower>().rankUpdate(m1.col(c),s1)._expression()),
                   ((s1 * m1.col(c) * m1.col(c).adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
                   
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Upper>().rankUpdate(m1.col(c),s1)._expression()),
                   ((s1 * m1.col(c) * m1.col(c).adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));
  rm2.setZero();
  VERIFY_IS_APPROX((rm2.template selfadjointView<Upper>().rankUpdate(m1.col(c),s1)._expression()),
                   ((s1 * m1.col(c) * m1.col(c).adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));
  m2.setZero();
  VERIFY_IS_APPROX((m2.template triangularView<Upper>() += s1 * m3.col(c) * m1.col(c).adjoint()).nestedExpression(),
                   ((s1 * m3.col(c) * m1.col(c).adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));
  rm2.setZero();
  VERIFY_IS_APPROX((rm2.template triangularView<Upper>() += s1 * m1.col(c) * m3.col(c).adjoint()).nestedExpression(),
                   ((s1 * m1.col(c) * m3.col(c).adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));
  
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Lower>().rankUpdate(m1.col(c).conjugate(),s1)._expression()),
                   ((s1 * m1.col(c).conjugate() * m1.col(c).conjugate().adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
                   
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Upper>().rankUpdate(m1.col(c).conjugate(),s1)._expression()),
                   ((s1 * m1.col(c).conjugate() * m1.col(c).conjugate().adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));
  
  
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Lower>().rankUpdate(m1.row(c),s1)._expression()),
                   ((s1 * m1.row(c).transpose() * m1.row(c).transpose().adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  rm2.setZero();
  VERIFY_IS_APPROX((rm2.template selfadjointView<Lower>().rankUpdate(m1.row(c),s1)._expression()),
                   ((s1 * m1.row(c).transpose() * m1.row(c).transpose().adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  m2.setZero();
  VERIFY_IS_APPROX((m2.template triangularView<Lower>() += s1 * m3.row(c).transpose() * m1.row(c).transpose().adjoint()).nestedExpression(),
                   ((s1 * m3.row(c).transpose() * m1.row(c).transpose().adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  rm2.setZero();
  VERIFY_IS_APPROX((rm2.template triangularView<Lower>() += s1 * m3.row(c).transpose() * m1.row(c).transpose().adjoint()).nestedExpression(),
                   ((s1 * m3.row(c).transpose() * m1.row(c).transpose().adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  
  
  m2.setZero();
  VERIFY_IS_APPROX((m2.template selfadjointView<Upper>().rankUpdate(m1.row(c).adjoint(),s1)._expression()),
                   ((s1 * m1.row(c).adjoint() * m1.row(c).adjoint().adjoint()).eval().template triangularView<Upper>().toDenseMatrix()));

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*rows,2*cols);
    Map<MatrixType,0,Stride<Dynamic,2> > map1(buffer.data(),rows,cols,Stride<Dynamic,2>(2*rows,2));
    buffer.setZero();
    VERIFY_IS_APPROX((map1.template selfadjointView<Lower>().rankUpdate(rhs2,s1)._expression()),
                      ((s1 * rhs2 * rhs2.adjoint()).eval().template triangularView<Lower>().toDenseMatrix()));
  }
}

void test_product_syrk()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    int s;
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_1( syrk(MatrixXf(s, s)) );
    CALL_SUBTEST_2( syrk(MatrixXd(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
    
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2);
    CALL_SUBTEST_3( syrk(MatrixXcf(s, s)) );
    CALL_SUBTEST_4( syrk(MatrixXcd(s, s)) );
    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }
}
