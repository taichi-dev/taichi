// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

float *ptr;
const float *const_ptr;

template<typename PlainObjectType,
         bool IsDynamicSize = PlainObjectType::SizeAtCompileTime == Dynamic,
         bool IsVector = PlainObjectType::IsVectorAtCompileTime
>
struct mapstaticmethods_impl {};

template<typename PlainObjectType, bool IsVector>
struct mapstaticmethods_impl<PlainObjectType, false, IsVector>
{
  static void run(const PlainObjectType& m)
  {
    mapstaticmethods_impl<PlainObjectType, true, IsVector>::run(m);

    int i = internal::random<int>(2,5), j = internal::random<int>(2,5);

    PlainObjectType::Map(ptr).setZero();
    PlainObjectType::MapAligned(ptr).setZero();
    PlainObjectType::Map(const_ptr).sum();
    PlainObjectType::MapAligned(const_ptr).sum();

    PlainObjectType::Map(ptr, InnerStride<>(i)).setZero();
    PlainObjectType::MapAligned(ptr, InnerStride<>(i)).setZero();
    PlainObjectType::Map(const_ptr, InnerStride<>(i)).sum();
    PlainObjectType::MapAligned(const_ptr, InnerStride<>(i)).sum();

    PlainObjectType::Map(ptr, InnerStride<2>()).setZero();
    PlainObjectType::MapAligned(ptr, InnerStride<3>()).setZero();
    PlainObjectType::Map(const_ptr, InnerStride<4>()).sum();
    PlainObjectType::MapAligned(const_ptr, InnerStride<5>()).sum();

    PlainObjectType::Map(ptr, OuterStride<>(i)).setZero();
    PlainObjectType::MapAligned(ptr, OuterStride<>(i)).setZero();
    PlainObjectType::Map(const_ptr, OuterStride<>(i)).sum();
    PlainObjectType::MapAligned(const_ptr, OuterStride<>(i)).sum();

    PlainObjectType::Map(ptr, OuterStride<2>()).setZero();
    PlainObjectType::MapAligned(ptr, OuterStride<3>()).setZero();
    PlainObjectType::Map(const_ptr, OuterStride<4>()).sum();
    PlainObjectType::MapAligned(const_ptr, OuterStride<5>()).sum();

    PlainObjectType::Map(ptr, Stride<Dynamic, Dynamic>(i,j)).setZero();
    PlainObjectType::MapAligned(ptr, Stride<2,Dynamic>(2,i)).setZero();
    PlainObjectType::Map(const_ptr, Stride<Dynamic,3>(i,3)).sum();
    PlainObjectType::MapAligned(const_ptr, Stride<Dynamic, Dynamic>(i,j)).sum();

    PlainObjectType::Map(ptr, Stride<2,3>()).setZero();
    PlainObjectType::MapAligned(ptr, Stride<3,4>()).setZero();
    PlainObjectType::Map(const_ptr, Stride<2,4>()).sum();
    PlainObjectType::MapAligned(const_ptr, Stride<5,3>()).sum();
  }
};

template<typename PlainObjectType>
struct mapstaticmethods_impl<PlainObjectType, true, false>
{
  static void run(const PlainObjectType& m)
  {
    Index rows = m.rows(), cols = m.cols();

    int i = internal::random<int>(2,5), j = internal::random<int>(2,5);

    PlainObjectType::Map(ptr, rows, cols).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols).setZero();
    PlainObjectType::Map(const_ptr, rows, cols).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols).sum();

    PlainObjectType::Map(ptr, rows, cols, InnerStride<>(i)).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, InnerStride<>(i)).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, InnerStride<>(i)).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, InnerStride<>(i)).sum();

    PlainObjectType::Map(ptr, rows, cols, InnerStride<2>()).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, InnerStride<3>()).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, InnerStride<4>()).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, InnerStride<5>()).sum();

    PlainObjectType::Map(ptr, rows, cols, OuterStride<>(i)).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, OuterStride<>(i)).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, OuterStride<>(i)).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, OuterStride<>(i)).sum();

    PlainObjectType::Map(ptr, rows, cols, OuterStride<2>()).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, OuterStride<3>()).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, OuterStride<4>()).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, OuterStride<5>()).sum();

    PlainObjectType::Map(ptr, rows, cols, Stride<Dynamic, Dynamic>(i,j)).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, Stride<2,Dynamic>(2,i)).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, Stride<Dynamic,3>(i,3)).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, Stride<Dynamic, Dynamic>(i,j)).sum();

    PlainObjectType::Map(ptr, rows, cols, Stride<2,3>()).setZero();
    PlainObjectType::MapAligned(ptr, rows, cols, Stride<3,4>()).setZero();
    PlainObjectType::Map(const_ptr, rows, cols, Stride<2,4>()).sum();
    PlainObjectType::MapAligned(const_ptr, rows, cols, Stride<5,3>()).sum();
  }
};

template<typename PlainObjectType>
struct mapstaticmethods_impl<PlainObjectType, true, true>
{
  static void run(const PlainObjectType& v)
  {
    Index size = v.size();

    int i = internal::random<int>(2,5);

    PlainObjectType::Map(ptr, size).setZero();
    PlainObjectType::MapAligned(ptr, size).setZero();
    PlainObjectType::Map(const_ptr, size).sum();
    PlainObjectType::MapAligned(const_ptr, size).sum();

    PlainObjectType::Map(ptr, size, InnerStride<>(i)).setZero();
    PlainObjectType::MapAligned(ptr, size, InnerStride<>(i)).setZero();
    PlainObjectType::Map(const_ptr, size, InnerStride<>(i)).sum();
    PlainObjectType::MapAligned(const_ptr, size, InnerStride<>(i)).sum();

    PlainObjectType::Map(ptr, size, InnerStride<2>()).setZero();
    PlainObjectType::MapAligned(ptr, size, InnerStride<3>()).setZero();
    PlainObjectType::Map(const_ptr, size, InnerStride<4>()).sum();
    PlainObjectType::MapAligned(const_ptr, size, InnerStride<5>()).sum();
  }
};

template<typename PlainObjectType>
void mapstaticmethods(const PlainObjectType& m)
{
  mapstaticmethods_impl<PlainObjectType>::run(m);
  VERIFY(true); // just to avoid 'unused function' warning
}

void test_mapstaticmethods()
{
  ptr = internal::aligned_new<float>(1000);
  for(int i = 0; i < 1000; i++) ptr[i] = float(i);

  const_ptr = ptr;

  CALL_SUBTEST_1(( mapstaticmethods(Matrix<float, 1, 1>()) ));
  CALL_SUBTEST_1(( mapstaticmethods(Vector2f()) ));
  CALL_SUBTEST_2(( mapstaticmethods(Vector3f()) ));
  CALL_SUBTEST_2(( mapstaticmethods(Matrix2f()) ));
  CALL_SUBTEST_3(( mapstaticmethods(Matrix4f()) ));
  CALL_SUBTEST_3(( mapstaticmethods(Array4f()) ));
  CALL_SUBTEST_4(( mapstaticmethods(Array3f()) ));
  CALL_SUBTEST_4(( mapstaticmethods(Array33f()) ));
  CALL_SUBTEST_5(( mapstaticmethods(Array44f()) ));
  CALL_SUBTEST_5(( mapstaticmethods(VectorXf(1)) ));
  CALL_SUBTEST_5(( mapstaticmethods(VectorXf(8)) ));
  CALL_SUBTEST_6(( mapstaticmethods(MatrixXf(1,1)) ));
  CALL_SUBTEST_6(( mapstaticmethods(MatrixXf(5,7)) ));
  CALL_SUBTEST_7(( mapstaticmethods(ArrayXf(1)) ));
  CALL_SUBTEST_7(( mapstaticmethods(ArrayXf(5)) ));
  CALL_SUBTEST_8(( mapstaticmethods(ArrayXXf(1,1)) ));
  CALL_SUBTEST_8(( mapstaticmethods(ArrayXXf(8,6)) ));

  internal::aligned_delete(ptr, 1000);
}

