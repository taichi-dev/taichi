// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>

template<typename Derived1, typename Derived2>
bool areNotApprox(const MatrixBase<Derived1>& m1, const MatrixBase<Derived2>& m2, typename Derived1::RealScalar epsilon = NumTraits<typename Derived1::RealScalar>::dummy_precision())
{
  return !((m1-m2).cwiseAbs2().maxCoeff() < epsilon * epsilon
                          * (std::max)(m1.cwiseAbs2().maxCoeff(), m2.cwiseAbs2().maxCoeff()));
}

template<typename MatrixType> void product(const MatrixType& m)
{
  /* this test covers the following files:
     Identity.h Product.h
  */
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> RowVectorType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> RowSquareMatrixType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> ColSquareMatrixType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime,
                         MatrixType::Flags&RowMajorBit?ColMajor:RowMajor> OtherMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);
  RowSquareMatrixType
             identity = RowSquareMatrixType::Identity(rows, rows),
             square = RowSquareMatrixType::Random(rows, rows),
             res = RowSquareMatrixType::Random(rows, rows);
  ColSquareMatrixType
             square2 = ColSquareMatrixType::Random(cols, cols),
             res2 = ColSquareMatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = internal::random<Scalar>();

  Index r  = internal::random<Index>(0, rows-1),
        c  = internal::random<Index>(0, cols-1),
        c2 = internal::random<Index>(0, cols-1);

  // begin testing Product.h: only associativity for now
  // (we use Transpose.h but this doesn't count as a test for it)
  VERIFY_IS_APPROX((m1*m1.transpose())*m2,  m1*(m1.transpose()*m2));
  m3 = m1;
  m3 *= m1.transpose() * m2;
  VERIFY_IS_APPROX(m3,                      m1 * (m1.transpose()*m2));
  VERIFY_IS_APPROX(m3,                      m1 * (m1.transpose()*m2));

  // continue testing Product.h: distributivity
  VERIFY_IS_APPROX(square*(m1 + m2),        square*m1+square*m2);
  VERIFY_IS_APPROX(square*(m1 - m2),        square*m1-square*m2);

  // continue testing Product.h: compatibility with ScalarMultiple.h
  VERIFY_IS_APPROX(s1*(square*m1),          (s1*square)*m1);
  VERIFY_IS_APPROX(s1*(square*m1),          square*(m1*s1));

  // test Product.h together with Identity.h
  VERIFY_IS_APPROX(v1,                      identity*v1);
  VERIFY_IS_APPROX(v1.transpose(),          v1.transpose() * identity);
  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX(MatrixType::Identity(rows, cols)(r,c), static_cast<Scalar>(r==c));

  if (rows!=cols)
     VERIFY_RAISES_ASSERT(m3 = m1*m1);

  // test the previous tests were not screwed up because operator* returns 0
  // (we use the more accurate default epsilon)
  if (!NumTraits<Scalar>::IsInteger && (std::min)(rows,cols)>1)
  {
    VERIFY(areNotApprox(m1.transpose()*m2,m2.transpose()*m1));
  }

  // test optimized operator+= path
  res = square;
  res.noalias() += m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square + m1 * m2.transpose());
  if (!NumTraits<Scalar>::IsInteger && (std::min)(rows,cols)>1)
  {
    VERIFY(areNotApprox(res,square + m2 * m1.transpose()));
  }
  vcres = vc2;
  vcres.noalias() += m1.transpose() * v1;
  VERIFY_IS_APPROX(vcres, vc2 + m1.transpose() * v1);

  // test optimized operator-= path
  res = square;
  res.noalias() -= m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square - (m1 * m2.transpose()));
  if (!NumTraits<Scalar>::IsInteger && (std::min)(rows,cols)>1)
  {
    VERIFY(areNotApprox(res,square - m2 * m1.transpose()));
  }
  vcres = vc2;
  vcres.noalias() -= m1.transpose() * v1;
  VERIFY_IS_APPROX(vcres, vc2 - m1.transpose() * v1);

  // test scaled products
  res = square;
  res.noalias() = s1 * m1 * m2.transpose();
  VERIFY_IS_APPROX(res, ((s1*m1).eval() * m2.transpose()));
  res = square;
  res.noalias() += s1 * m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square + ((s1*m1).eval() * m2.transpose()));
  res = square;
  res.noalias() -= s1 * m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square - ((s1*m1).eval() * m2.transpose()));

  // test d ?= a+b*c rules
  res.noalias() = square + m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square + m1 * m2.transpose());
  res.noalias() += square + m1 * m2.transpose();
  VERIFY_IS_APPROX(res, 2*(square + m1 * m2.transpose()));
  res.noalias() -= square + m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square + m1 * m2.transpose());

  // test d ?= a-b*c rules
  res.noalias() = square - m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square - m1 * m2.transpose());
  res.noalias() += square - m1 * m2.transpose();
  VERIFY_IS_APPROX(res, 2*(square - m1 * m2.transpose()));
  res.noalias() -= square - m1 * m2.transpose();
  VERIFY_IS_APPROX(res, square - m1 * m2.transpose());


  tm1 = m1;
  VERIFY_IS_APPROX(tm1.transpose() * v1, m1.transpose() * v1);
  VERIFY_IS_APPROX(v1.transpose() * tm1, v1.transpose() * m1);

  // test submatrix and matrix/vector product
  for (int i=0; i<rows; ++i)
    res.row(i) = m1.row(i) * m2.transpose();
  VERIFY_IS_APPROX(res, m1 * m2.transpose());
  // the other way round:
  for (int i=0; i<rows; ++i)
    res.col(i) = m1 * m2.transpose().col(i);
  VERIFY_IS_APPROX(res, m1 * m2.transpose());

  res2 = square2;
  res2.noalias() += m1.transpose() * m2;
  VERIFY_IS_APPROX(res2, square2 + m1.transpose() * m2);
  if (!NumTraits<Scalar>::IsInteger && (std::min)(rows,cols)>1)
  {
    VERIFY(areNotApprox(res2,square2 + m2.transpose() * m1));
  }

  VERIFY_IS_APPROX(res.col(r).noalias() = square.adjoint() * square.col(r), (square.adjoint() * square.col(r)).eval());
  VERIFY_IS_APPROX(res.col(r).noalias() = square * square.col(r), (square * square.col(r)).eval());

  // vector at runtime (see bug 1166)
  {
    RowSquareMatrixType ref(square);
    ColSquareMatrixType ref2(square2);
    ref = res = square;
    VERIFY_IS_APPROX(res.block(0,0,1,rows).noalias() = m1.col(0).transpose() * square.transpose(),            (ref.row(0) = m1.col(0).transpose() * square.transpose()));
    VERIFY_IS_APPROX(res.block(0,0,1,rows).noalias() = m1.block(0,0,rows,1).transpose() * square.transpose(), (ref.row(0) = m1.col(0).transpose() * square.transpose()));
    VERIFY_IS_APPROX(res.block(0,0,1,rows).noalias() = m1.col(0).transpose() * square,                        (ref.row(0) = m1.col(0).transpose() * square));
    VERIFY_IS_APPROX(res.block(0,0,1,rows).noalias() = m1.block(0,0,rows,1).transpose() * square,             (ref.row(0) = m1.col(0).transpose() * square));
    ref2 = res2 = square2;
    VERIFY_IS_APPROX(res2.block(0,0,1,cols).noalias() = m1.row(0) * square2.transpose(),                      (ref2.row(0) = m1.row(0) * square2.transpose()));
    VERIFY_IS_APPROX(res2.block(0,0,1,cols).noalias() = m1.block(0,0,1,cols) * square2.transpose(),           (ref2.row(0) = m1.row(0) * square2.transpose()));
    VERIFY_IS_APPROX(res2.block(0,0,1,cols).noalias() = m1.row(0) * square2,                                  (ref2.row(0) = m1.row(0) * square2));
    VERIFY_IS_APPROX(res2.block(0,0,1,cols).noalias() = m1.block(0,0,1,cols) * square2,                       (ref2.row(0) = m1.row(0) * square2));
  }

  // vector.block() (see bug 1283)
  {
    RowVectorType w1(rows);
    VERIFY_IS_APPROX(square * v1.block(0,0,rows,1), square * v1);
    VERIFY_IS_APPROX(w1.noalias() = square * v1.block(0,0,rows,1), square * v1);
    VERIFY_IS_APPROX(w1.block(0,0,rows,1).noalias() = square * v1.block(0,0,rows,1), square * v1);

    Matrix<Scalar,1,MatrixType::ColsAtCompileTime> w2(cols);
    VERIFY_IS_APPROX(vc2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.noalias() = vc2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.block(0,0,1,cols).noalias() = vc2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);

    vc2 = square2.block(0,0,1,cols).transpose();
    VERIFY_IS_APPROX(square2.block(0,0,1,cols) * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.noalias() = square2.block(0,0,1,cols) * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.block(0,0,1,cols).noalias() = square2.block(0,0,1,cols) * square2, vc2.transpose() * square2);

    vc2 = square2.block(0,0,cols,1);
    VERIFY_IS_APPROX(square2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.noalias() = square2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);
    VERIFY_IS_APPROX(w2.block(0,0,1,cols).noalias() = square2.block(0,0,cols,1).transpose() * square2, vc2.transpose() * square2);
  }

  // inner product
  {
    Scalar x = square2.row(c) * square2.col(c2);
    VERIFY_IS_APPROX(x, square2.row(c).transpose().cwiseProduct(square2.col(c2)).sum());
  }

  // outer product
  {
    VERIFY_IS_APPROX(m1.col(c) * m1.row(r), m1.block(0,c,rows,1) * m1.block(r,0,1,cols));
    VERIFY_IS_APPROX(m1.row(r).transpose() * m1.col(c).transpose(), m1.block(r,0,1,cols).transpose() * m1.block(0,c,rows,1).transpose());
    VERIFY_IS_APPROX(m1.block(0,c,rows,1) * m1.row(r), m1.block(0,c,rows,1) * m1.block(r,0,1,cols));
    VERIFY_IS_APPROX(m1.col(c) * m1.block(r,0,1,cols), m1.block(0,c,rows,1) * m1.block(r,0,1,cols));
    VERIFY_IS_APPROX(m1.leftCols(1) * m1.row(r), m1.block(0,0,rows,1) * m1.block(r,0,1,cols));
    VERIFY_IS_APPROX(m1.col(c) * m1.topRows(1), m1.block(0,c,rows,1) * m1.block(0,0,1,cols));
  }

  // Aliasing
  {
    ColVectorType x(cols); x.setRandom();
    ColVectorType z(x);
    ColVectorType y(cols); y.setZero();
    ColSquareMatrixType A(cols,cols); A.setRandom();
    // CwiseBinaryOp
    VERIFY_IS_APPROX(x = y + A*x, A*z);
    x = z;
    // CwiseUnaryOp
    VERIFY_IS_APPROX(x = Scalar(1.)*(A*x), A*z);
  }

  // regression for blas_trais
  {
    VERIFY_IS_APPROX(square * (square*square).transpose(), square * square.transpose() * square.transpose());
    VERIFY_IS_APPROX(square * (-(square*square)), -square * square * square);
    VERIFY_IS_APPROX(square * (s1*(square*square)), s1 * square * square * square);
    VERIFY_IS_APPROX(square * (square*square).conjugate(), square * square.conjugate() * square.conjugate());
  }

  // destination with a non-default inner-stride
  // see bug 1741
  if(!MatrixType::IsRowMajor)
  {
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*rows,2*rows);
    Map<RowSquareMatrixType,0,Stride<Dynamic,2> > map1(buffer.data(),rows,rows,Stride<Dynamic,2>(2*rows,2));
    buffer.setZero();
    VERIFY_IS_APPROX(map1 = m1 * m2.transpose(), (m1 * m2.transpose()).eval());
    buffer.setZero();
    VERIFY_IS_APPROX(map1.noalias() = m1 * m2.transpose(), (m1 * m2.transpose()).eval());
    buffer.setZero();
    VERIFY_IS_APPROX(map1.noalias() += m1 * m2.transpose(), (m1 * m2.transpose()).eval());
  }

}
