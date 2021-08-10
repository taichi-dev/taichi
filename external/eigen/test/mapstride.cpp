// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<int Alignment,typename VectorType> void map_class_vector(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;

  Index size = m.size();

  VectorType v = VectorType::Random(size);

  Index arraysize = 3*size;
  
  Scalar* a_array = internal::aligned_new<Scalar>(arraysize+1);
  Scalar* array = a_array;
  if(Alignment!=Aligned)
    array = (Scalar*)(internal::IntPtr(a_array) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));

  {
    Map<VectorType, Alignment, InnerStride<3> > map(array, size);
    map = v;
    for(int i = 0; i < size; ++i)
    {
      VERIFY(array[3*i] == v[i]);
      VERIFY(map[i] == v[i]);
    }
  }

  {
    Map<VectorType, Unaligned, InnerStride<Dynamic> > map(array, size, InnerStride<Dynamic>(2));
    map = v;
    for(int i = 0; i < size; ++i)
    {
      VERIFY(array[2*i] == v[i]);
      VERIFY(map[i] == v[i]);
    }
  }

  internal::aligned_delete(a_array, arraysize+1);
}

template<int Alignment,typename MatrixType> void map_class_matrix(const MatrixType& _m)
{
  typedef typename MatrixType::Scalar Scalar;

  Index rows = _m.rows(), cols = _m.cols();

  MatrixType m = MatrixType::Random(rows,cols);
  Scalar s1 = internal::random<Scalar>();

  Index arraysize = 4*(rows+4)*(cols+4);

  Scalar* a_array1 = internal::aligned_new<Scalar>(arraysize+1);
  Scalar* array1 = a_array1;
  if(Alignment!=Aligned)
    array1 = (Scalar*)(internal::IntPtr(a_array1) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));

  Scalar a_array2[256];
  Scalar* array2 = a_array2;
  if(Alignment!=Aligned)
    array2 = (Scalar*)(internal::IntPtr(a_array2) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));
  else
    array2 = (Scalar*)(((internal::UIntPtr(a_array2)+EIGEN_MAX_ALIGN_BYTES-1)/EIGEN_MAX_ALIGN_BYTES)*EIGEN_MAX_ALIGN_BYTES);
  Index maxsize2 = a_array2 - array2 + 256;
  
  // test no inner stride and some dynamic outer stride
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (m.innerSize()+1)*m.outerSize() > maxsize2)
      break;
    Scalar* array = (k==0 ? array1 : array2);
    
    Map<MatrixType, Alignment, OuterStride<Dynamic> > map(array, rows, cols, OuterStride<Dynamic>(m.innerSize()+1));
    map = m;
    VERIFY(map.outerStride() == map.innerSize()+1);
    for(int i = 0; i < m.outerSize(); ++i)
      for(int j = 0; j < m.innerSize(); ++j)
      {
        VERIFY(array[map.outerStride()*i+j] == m.coeffByOuterInner(i,j));
        VERIFY(map.coeffByOuterInner(i,j) == m.coeffByOuterInner(i,j));
      }
    VERIFY_IS_APPROX(s1*map,s1*m);
    map *= s1;
    VERIFY_IS_APPROX(map,s1*m);
  }

  // test no inner stride and an outer stride of +4. This is quite important as for fixed-size matrices,
  // this allows to hit the special case where it's vectorizable.
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (m.innerSize()+4)*m.outerSize() > maxsize2)
      break;
    Scalar* array = (k==0 ? array1 : array2);
    
    enum {
      InnerSize = MatrixType::InnerSizeAtCompileTime,
      OuterStrideAtCompileTime = InnerSize==Dynamic ? Dynamic : InnerSize+4
    };
    Map<MatrixType, Alignment, OuterStride<OuterStrideAtCompileTime> >
      map(array, rows, cols, OuterStride<OuterStrideAtCompileTime>(m.innerSize()+4));
    map = m;
    VERIFY(map.outerStride() == map.innerSize()+4);
    for(int i = 0; i < m.outerSize(); ++i)
      for(int j = 0; j < m.innerSize(); ++j)
      {
        VERIFY(array[map.outerStride()*i+j] == m.coeffByOuterInner(i,j));
        VERIFY(map.coeffByOuterInner(i,j) == m.coeffByOuterInner(i,j));
      }
    VERIFY_IS_APPROX(s1*map,s1*m);
    map *= s1;
    VERIFY_IS_APPROX(map,s1*m);
  }

  // test both inner stride and outer stride
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (2*m.innerSize()+1)*(m.outerSize()*2) > maxsize2)
      break;
    Scalar* array = (k==0 ? array1 : array2);
    
    Map<MatrixType, Alignment, Stride<Dynamic,Dynamic> > map(array, rows, cols, Stride<Dynamic,Dynamic>(2*m.innerSize()+1, 2));
    map = m;
    VERIFY(map.outerStride() == 2*map.innerSize()+1);
    VERIFY(map.innerStride() == 2);
    for(int i = 0; i < m.outerSize(); ++i)
      for(int j = 0; j < m.innerSize(); ++j)
      {
        VERIFY(array[map.outerStride()*i+map.innerStride()*j] == m.coeffByOuterInner(i,j));
        VERIFY(map.coeffByOuterInner(i,j) == m.coeffByOuterInner(i,j));
      }
    VERIFY_IS_APPROX(s1*map,s1*m);
    map *= s1;
    VERIFY_IS_APPROX(map,s1*m);
  }

  // test inner stride and no outer stride
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (m.innerSize()*2)*m.outerSize() > maxsize2)
      break;
    Scalar* array = (k==0 ? array1 : array2);

    Map<MatrixType, Alignment, InnerStride<Dynamic> > map(array, rows, cols, InnerStride<Dynamic>(2));
    map = m;
    VERIFY(map.outerStride() == map.innerSize()*2);
    for(int i = 0; i < m.outerSize(); ++i)
      for(int j = 0; j < m.innerSize(); ++j)
      {
        VERIFY(array[map.innerSize()*i*2+j*2] == m.coeffByOuterInner(i,j));
        VERIFY(map.coeffByOuterInner(i,j) == m.coeffByOuterInner(i,j));
      }
    VERIFY_IS_APPROX(s1*map,s1*m);
    map *= s1;
    VERIFY_IS_APPROX(map,s1*m);
  }

  internal::aligned_delete(a_array1, arraysize+1);
}

// Additional tests for inner-stride but no outer-stride
template<int>
void bug1453()
{
  const int data[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  typedef Matrix<int,Dynamic,Dynamic,RowMajor> RowMatrixXi;
  typedef Matrix<int,2,3,ColMajor> ColMatrix23i;
  typedef Matrix<int,3,2,ColMajor> ColMatrix32i;
  typedef Matrix<int,2,3,RowMajor> RowMatrix23i;
  typedef Matrix<int,3,2,RowMajor> RowMatrix32i;

  VERIFY_IS_APPROX(MatrixXi::Map(data, 2, 3, InnerStride<2>()), MatrixXi::Map(data, 2, 3, Stride<4,2>()));
  VERIFY_IS_APPROX(MatrixXi::Map(data, 2, 3, InnerStride<>(2)), MatrixXi::Map(data, 2, 3, Stride<4,2>()));
  VERIFY_IS_APPROX(MatrixXi::Map(data, 3, 2, InnerStride<2>()), MatrixXi::Map(data, 3, 2, Stride<6,2>()));
  VERIFY_IS_APPROX(MatrixXi::Map(data, 3, 2, InnerStride<>(2)), MatrixXi::Map(data, 3, 2, Stride<6,2>()));

  VERIFY_IS_APPROX(RowMatrixXi::Map(data, 2, 3, InnerStride<2>()), RowMatrixXi::Map(data, 2, 3, Stride<6,2>()));
  VERIFY_IS_APPROX(RowMatrixXi::Map(data, 2, 3, InnerStride<>(2)), RowMatrixXi::Map(data, 2, 3, Stride<6,2>()));
  VERIFY_IS_APPROX(RowMatrixXi::Map(data, 3, 2, InnerStride<2>()), RowMatrixXi::Map(data, 3, 2, Stride<4,2>()));
  VERIFY_IS_APPROX(RowMatrixXi::Map(data, 3, 2, InnerStride<>(2)), RowMatrixXi::Map(data, 3, 2, Stride<4,2>()));

  VERIFY_IS_APPROX(ColMatrix23i::Map(data, InnerStride<2>()), MatrixXi::Map(data, 2, 3, Stride<4,2>()));
  VERIFY_IS_APPROX(ColMatrix23i::Map(data, InnerStride<>(2)), MatrixXi::Map(data, 2, 3, Stride<4,2>()));
  VERIFY_IS_APPROX(ColMatrix32i::Map(data, InnerStride<2>()), MatrixXi::Map(data, 3, 2, Stride<6,2>()));
  VERIFY_IS_APPROX(ColMatrix32i::Map(data, InnerStride<>(2)), MatrixXi::Map(data, 3, 2, Stride<6,2>()));

  VERIFY_IS_APPROX(RowMatrix23i::Map(data, InnerStride<2>()), RowMatrixXi::Map(data, 2, 3, Stride<6,2>()));
  VERIFY_IS_APPROX(RowMatrix23i::Map(data, InnerStride<>(2)), RowMatrixXi::Map(data, 2, 3, Stride<6,2>()));
  VERIFY_IS_APPROX(RowMatrix32i::Map(data, InnerStride<2>()), RowMatrixXi::Map(data, 3, 2, Stride<4,2>()));
  VERIFY_IS_APPROX(RowMatrix32i::Map(data, InnerStride<>(2)), RowMatrixXi::Map(data, 3, 2, Stride<4,2>()));
}

void test_mapstride()
{
  for(int i = 0; i < g_repeat; i++) {
    int maxn = 30;
    CALL_SUBTEST_1( map_class_vector<Aligned>(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( map_class_vector<Unaligned>(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( map_class_vector<Aligned>(Vector4d()) );
    CALL_SUBTEST_2( map_class_vector<Unaligned>(Vector4d()) );
    CALL_SUBTEST_3( map_class_vector<Aligned>(RowVector4f()) );
    CALL_SUBTEST_3( map_class_vector<Unaligned>(RowVector4f()) );
    CALL_SUBTEST_4( map_class_vector<Aligned>(VectorXcf(internal::random<int>(1,maxn))) );
    CALL_SUBTEST_4( map_class_vector<Unaligned>(VectorXcf(internal::random<int>(1,maxn))) );
    CALL_SUBTEST_5( map_class_vector<Aligned>(VectorXi(internal::random<int>(1,maxn))) );
    CALL_SUBTEST_5( map_class_vector<Unaligned>(VectorXi(internal::random<int>(1,maxn))) );

    CALL_SUBTEST_1( map_class_matrix<Aligned>(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_1( map_class_matrix<Unaligned>(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( map_class_matrix<Aligned>(Matrix4d()) );
    CALL_SUBTEST_2( map_class_matrix<Unaligned>(Matrix4d()) );
    CALL_SUBTEST_3( map_class_matrix<Aligned>(Matrix<float,3,5>()) );
    CALL_SUBTEST_3( map_class_matrix<Unaligned>(Matrix<float,3,5>()) );
    CALL_SUBTEST_3( map_class_matrix<Aligned>(Matrix<float,4,8>()) );
    CALL_SUBTEST_3( map_class_matrix<Unaligned>(Matrix<float,4,8>()) );
    CALL_SUBTEST_4( map_class_matrix<Aligned>(MatrixXcf(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );
    CALL_SUBTEST_4( map_class_matrix<Unaligned>(MatrixXcf(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );
    CALL_SUBTEST_5( map_class_matrix<Aligned>(MatrixXi(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );
    CALL_SUBTEST_5( map_class_matrix<Unaligned>(MatrixXi(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );
    CALL_SUBTEST_6( map_class_matrix<Aligned>(MatrixXcd(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );
    CALL_SUBTEST_6( map_class_matrix<Unaligned>(MatrixXcd(internal::random<int>(1,maxn),internal::random<int>(1,maxn))) );

    CALL_SUBTEST_5( bug1453<0>() );
    
    TEST_SET_BUT_UNUSED_VARIABLE(maxn);
  }
}
