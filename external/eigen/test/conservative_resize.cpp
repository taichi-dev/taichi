// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Storage>
void run_matrix_tests()
{
  typedef Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Storage> MatrixType;

  MatrixType m, n;

  // boundary cases ...
  m = n = MatrixType::Random(50,50);
  m.conservativeResize(1,50);
  VERIFY_IS_APPROX(m, n.block(0,0,1,50));

  m = n = MatrixType::Random(50,50);
  m.conservativeResize(50,1);
  VERIFY_IS_APPROX(m, n.block(0,0,50,1));

  m = n = MatrixType::Random(50,50);
  m.conservativeResize(50,50);
  VERIFY_IS_APPROX(m, n.block(0,0,50,50));

  // random shrinking ...
  for (int i=0; i<25; ++i)
  {
    const Index rows = internal::random<Index>(1,50);
    const Index cols = internal::random<Index>(1,50);
    m = n = MatrixType::Random(50,50);
    m.conservativeResize(rows,cols);
    VERIFY_IS_APPROX(m, n.block(0,0,rows,cols));
  }

  // random growing with zeroing ...
  for (int i=0; i<25; ++i)
  {
    const Index rows = internal::random<Index>(50,75);
    const Index cols = internal::random<Index>(50,75);
    m = n = MatrixType::Random(50,50);
    m.conservativeResizeLike(MatrixType::Zero(rows,cols));
    VERIFY_IS_APPROX(m.block(0,0,n.rows(),n.cols()), n);
    VERIFY( rows<=50 || m.block(50,0,rows-50,cols).sum() == Scalar(0) );
    VERIFY( cols<=50 || m.block(0,50,rows,cols-50).sum() == Scalar(0) );
  }
}

template <typename Scalar>
void run_vector_tests()
{
  typedef Matrix<Scalar, 1, Eigen::Dynamic> VectorType;

  VectorType m, n;

  // boundary cases ...
  m = n = VectorType::Random(50);
  m.conservativeResize(1);
  VERIFY_IS_APPROX(m, n.segment(0,1));

  m = n = VectorType::Random(50);
  m.conservativeResize(50);
  VERIFY_IS_APPROX(m, n.segment(0,50));
  
  m = n = VectorType::Random(50);
  m.conservativeResize(m.rows(),1);
  VERIFY_IS_APPROX(m, n.segment(0,1));

  m = n = VectorType::Random(50);
  m.conservativeResize(m.rows(),50);
  VERIFY_IS_APPROX(m, n.segment(0,50));

  // random shrinking ...
  for (int i=0; i<50; ++i)
  {
    const int size = internal::random<int>(1,50);
    m = n = VectorType::Random(50);
    m.conservativeResize(size);
    VERIFY_IS_APPROX(m, n.segment(0,size));
    
    m = n = VectorType::Random(50);
    m.conservativeResize(m.rows(), size);
    VERIFY_IS_APPROX(m, n.segment(0,size));
  }

  // random growing with zeroing ...
  for (int i=0; i<50; ++i)
  {
    const int size = internal::random<int>(50,100);
    m = n = VectorType::Random(50);
    m.conservativeResizeLike(VectorType::Zero(size));
    VERIFY_IS_APPROX(m.segment(0,50), n);
    VERIFY( size<=50 || m.segment(50,size-50).sum() == Scalar(0) );
    
    m = n = VectorType::Random(50);
    m.conservativeResizeLike(Matrix<Scalar,Dynamic,Dynamic>::Zero(1,size));
    VERIFY_IS_APPROX(m.segment(0,50), n);
    VERIFY( size<=50 || m.segment(50,size-50).sum() == Scalar(0) );
  }
}

void test_conservative_resize()
{
  for(int i=0; i<g_repeat; ++i)
  {
    CALL_SUBTEST_1((run_matrix_tests<int, Eigen::RowMajor>()));
    CALL_SUBTEST_1((run_matrix_tests<int, Eigen::ColMajor>()));
    CALL_SUBTEST_2((run_matrix_tests<float, Eigen::RowMajor>()));
    CALL_SUBTEST_2((run_matrix_tests<float, Eigen::ColMajor>()));
    CALL_SUBTEST_3((run_matrix_tests<double, Eigen::RowMajor>()));
    CALL_SUBTEST_3((run_matrix_tests<double, Eigen::ColMajor>()));
    CALL_SUBTEST_4((run_matrix_tests<std::complex<float>, Eigen::RowMajor>()));
    CALL_SUBTEST_4((run_matrix_tests<std::complex<float>, Eigen::ColMajor>()));
    CALL_SUBTEST_5((run_matrix_tests<std::complex<double>, Eigen::RowMajor>()));
    CALL_SUBTEST_6((run_matrix_tests<std::complex<double>, Eigen::ColMajor>()));

    CALL_SUBTEST_1((run_vector_tests<int>()));
    CALL_SUBTEST_2((run_vector_tests<float>()));
    CALL_SUBTEST_3((run_vector_tests<double>()));
    CALL_SUBTEST_4((run_vector_tests<std::complex<float> >()));
    CALL_SUBTEST_5((run_vector_tests<std::complex<double> >()));
  }
}
