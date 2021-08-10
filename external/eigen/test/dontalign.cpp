// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined EIGEN_TEST_PART_1 || defined EIGEN_TEST_PART_2 || defined EIGEN_TEST_PART_3 || defined EIGEN_TEST_PART_4
#define EIGEN_DONT_ALIGN
#elif defined EIGEN_TEST_PART_5 || defined EIGEN_TEST_PART_6 || defined EIGEN_TEST_PART_7 || defined EIGEN_TEST_PART_8
#define EIGEN_DONT_ALIGN_STATICALLY
#endif

#include "main.h"
#include <Eigen/Dense>

template<typename MatrixType>
void dontalign(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType a = MatrixType::Random(rows,cols);
  SquareMatrixType square = SquareMatrixType::Random(rows,rows);
  VectorType v = VectorType::Random(rows);

  VERIFY_IS_APPROX(v, square * square.colPivHouseholderQr().solve(v));
  square = square.inverse().eval();
  a = square * a;
  square = square*square;
  v = square * v;
  v = a.adjoint() * v;
  VERIFY(square.determinant() != Scalar(0));

  // bug 219: MapAligned() was giving an assert with EIGEN_DONT_ALIGN, because Map Flags were miscomputed
  Scalar* array = internal::aligned_new<Scalar>(rows);
  v = VectorType::MapAligned(array, rows);
  internal::aligned_delete(array, rows);
}

void test_dontalign()
{
#if defined EIGEN_TEST_PART_1 || defined EIGEN_TEST_PART_5
  dontalign(Matrix3d());
  dontalign(Matrix4f());
#elif defined EIGEN_TEST_PART_2 || defined EIGEN_TEST_PART_6
  dontalign(Matrix3cd());
  dontalign(Matrix4cf());
#elif defined EIGEN_TEST_PART_3 || defined EIGEN_TEST_PART_7
  dontalign(Matrix<float, 32, 32>());
  dontalign(Matrix<std::complex<float>, 32, 32>());
#elif defined EIGEN_TEST_PART_4 || defined EIGEN_TEST_PART_8
  dontalign(MatrixXd(32, 32));
  dontalign(MatrixXcf(32, 32));
#endif
}
