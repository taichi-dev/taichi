// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void replicate(const MatrixType& m)
{
  /* this test covers the following files:
     Replicate.cpp
  */
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
  typedef Matrix<Scalar, Dynamic, 1> VectorX;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);

  VectorType v1 = VectorType::Random(rows);

  MatrixX x1, x2;
  VectorX vx1;

  int  f1 = internal::random<int>(1,10),
       f2 = internal::random<int>(1,10);

  x1.resize(rows*f1,cols*f2);
  for(int j=0; j<f2; j++)
  for(int i=0; i<f1; i++)
    x1.block(i*rows,j*cols,rows,cols) = m1;
  VERIFY_IS_APPROX(x1, m1.replicate(f1,f2));

  x2.resize(2*rows,3*cols);
  x2 << m2, m2, m2,
        m2, m2, m2;
  VERIFY_IS_APPROX(x2, (m2.template replicate<2,3>()));
  
  x2.resize(rows,3*cols);
  x2 << m2, m2, m2;
  VERIFY_IS_APPROX(x2, (m2.template replicate<1,3>()));
  
  vx1.resize(3*rows,cols);
  vx1 << m2, m2, m2;
  VERIFY_IS_APPROX(vx1+vx1, vx1+(m2.template replicate<3,1>()));
  
  vx1=m2+(m2.colwise().replicate(1));
  
  if(m2.cols()==1)
    VERIFY_IS_APPROX(m2.coeff(0), (m2.template replicate<3,1>().coeff(m2.rows())));

  x2.resize(rows,f1);
  for (int j=0; j<f1; ++j)
    x2.col(j) = v1;
  VERIFY_IS_APPROX(x2, v1.rowwise().replicate(f1));

  vx1.resize(rows*f2);
  for (int j=0; j<f2; ++j)
    vx1.segment(j*rows,rows) = v1;
  VERIFY_IS_APPROX(vx1, v1.colwise().replicate(f2));
}

void test_array_replicate()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( replicate(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( replicate(Vector2f()) );
    CALL_SUBTEST_3( replicate(Vector3d()) );
    CALL_SUBTEST_4( replicate(Vector4f()) );
    CALL_SUBTEST_5( replicate(VectorXf(16)) );
    CALL_SUBTEST_6( replicate(VectorXcd(10)) );
  }
}
