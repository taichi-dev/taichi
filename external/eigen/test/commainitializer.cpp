// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"


template<int M1, int M2, int N1, int N2>
void test_blocks()
{
  Matrix<int, M1+M2, N1+N2> m_fixed;
  MatrixXi m_dynamic(M1+M2, N1+N2);

  Matrix<int, M1, N1> mat11; mat11.setRandom();
  Matrix<int, M1, N2> mat12; mat12.setRandom();
  Matrix<int, M2, N1> mat21; mat21.setRandom();
  Matrix<int, M2, N2> mat22; mat22.setRandom();

  MatrixXi matx11 = mat11, matx12 = mat12, matx21 = mat21, matx22 = mat22;

  {
    VERIFY_IS_EQUAL((m_fixed << mat11, mat12, mat21, matx22).finished(), (m_dynamic << mat11, matx12, mat21, matx22).finished());
    VERIFY_IS_EQUAL((m_fixed.template topLeftCorner<M1,N1>()), mat11);
    VERIFY_IS_EQUAL((m_fixed.template topRightCorner<M1,N2>()), mat12);
    VERIFY_IS_EQUAL((m_fixed.template bottomLeftCorner<M2,N1>()), mat21);
    VERIFY_IS_EQUAL((m_fixed.template bottomRightCorner<M2,N2>()), mat22);
    VERIFY_IS_EQUAL((m_fixed << mat12, mat11, matx21, mat22).finished(), (m_dynamic << mat12, matx11, matx21, mat22).finished());
  }

  if(N1 > 0)
  {
    VERIFY_RAISES_ASSERT((m_fixed << mat11, mat12, mat11, mat21, mat22));
    VERIFY_RAISES_ASSERT((m_fixed << mat11, mat12, mat21, mat21, mat22));
  }
  else
  {
    // allow insertion of zero-column blocks:
    VERIFY_IS_EQUAL((m_fixed << mat11, mat12, mat11, mat11, mat21, mat21, mat22).finished(), (m_dynamic << mat12, mat22).finished());
  }
  if(M1 != M2)
  {
    VERIFY_RAISES_ASSERT((m_fixed << mat11, mat21, mat12, mat22));
  }
}


template<int N>
struct test_block_recursion
{
  static void run()
  {
    test_blocks<(N>>6)&3, (N>>4)&3, (N>>2)&3, N & 3>();
    test_block_recursion<N-1>::run();
  }
};

template<>
struct test_block_recursion<-1>
{
  static void run() { }
};

void test_commainitializer()
{
  Matrix3d m3;
  Matrix4d m4;

  VERIFY_RAISES_ASSERT( (m3 << 1, 2, 3, 4, 5, 6, 7, 8) );
  
  #ifndef _MSC_VER
  VERIFY_RAISES_ASSERT( (m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) );
  #endif

  double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Matrix3d ref = Map<Matrix<double,3,3,RowMajor> >(data);

  m3 = Matrix3d::Random();
  m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  VERIFY_IS_APPROX(m3, ref );

  Vector3d vec[3];
  vec[0] << 1, 4, 7;
  vec[1] << 2, 5, 8;
  vec[2] << 3, 6, 9;
  m3 = Matrix3d::Random();
  m3 << vec[0], vec[1], vec[2];
  VERIFY_IS_APPROX(m3, ref);

  vec[0] << 1, 2, 3;
  vec[1] << 4, 5, 6;
  vec[2] << 7, 8, 9;
  m3 = Matrix3d::Random();
  m3 << vec[0].transpose(),
        4, 5, 6,
        vec[2].transpose();
  VERIFY_IS_APPROX(m3, ref);


  // recursively test all block-sizes from 0 to 3:
  test_block_recursion<(1<<8) - 1>();
}
