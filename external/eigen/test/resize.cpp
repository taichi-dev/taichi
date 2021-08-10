// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<DenseIndex rows, DenseIndex cols>
void resizeLikeTest()
{
  MatrixXf A(rows, cols);
  MatrixXf B;
  Matrix<double, rows, cols> C;
  B.resizeLike(A);
  C.resizeLike(B);  // Shouldn't crash.
  VERIFY(B.rows() == rows && B.cols() == cols);

  VectorXf x(rows);
  RowVectorXf y;
  y.resizeLike(x);
  VERIFY(y.rows() == 1 && y.cols() == rows);

  y.resize(cols);
  x.resizeLike(y);
  VERIFY(x.rows() == cols && x.cols() == 1);
}

void resizeLikeTest12() { resizeLikeTest<1,2>(); }
void resizeLikeTest1020() { resizeLikeTest<10,20>(); }
void resizeLikeTest31() { resizeLikeTest<3,1>(); }

void test_resize()
{
  CALL_SUBTEST(resizeLikeTest12() );
  CALL_SUBTEST(resizeLikeTest1020() );
  CALL_SUBTEST(resizeLikeTest31() );
}
