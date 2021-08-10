// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_TEST_PART_1)
  // default
#elif defined(EIGEN_TEST_PART_2)
  #define EIGEN_MAX_STATIC_ALIGN_BYTES 16
  #define EIGEN_MAX_ALIGN_BYTES 16
#elif defined(EIGEN_TEST_PART_3)
  #define EIGEN_MAX_STATIC_ALIGN_BYTES 32
  #define EIGEN_MAX_ALIGN_BYTES 32
#elif defined(EIGEN_TEST_PART_4)
  #define EIGEN_MAX_STATIC_ALIGN_BYTES 64
  #define EIGEN_MAX_ALIGN_BYTES 64
#endif

#include "main.h"

typedef Matrix<float,  6,1> Vector6f;
typedef Matrix<float,  8,1> Vector8f;
typedef Matrix<float, 12,1> Vector12f;

typedef Matrix<double, 5,1> Vector5d;
typedef Matrix<double, 6,1> Vector6d;
typedef Matrix<double, 7,1> Vector7d;
typedef Matrix<double, 8,1> Vector8d;
typedef Matrix<double, 9,1> Vector9d;
typedef Matrix<double,10,1> Vector10d;
typedef Matrix<double,12,1> Vector12d;

struct TestNew1
{
  MatrixXd m; // good: m will allocate its own array, taking care of alignment.
  TestNew1() : m(20,20) {}
};

struct TestNew2
{
  Matrix3d m; // good: m's size isn't a multiple of 16 bytes, so m doesn't have to be 16-byte aligned,
              // 8-byte alignment is good enough here, which we'll get automatically
};

struct TestNew3
{
  Vector2f m; // good: m's size isn't a multiple of 16 bytes, so m doesn't have to be 16-byte aligned
};

struct TestNew4
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vector2d m;
  float f; // make the struct have sizeof%16!=0 to make it a little more tricky when we allow an array of 2 such objects
};

struct TestNew5
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float f; // try the f at first -- the EIGEN_ALIGN_MAX attribute of m should make that still work
  Matrix4f m;
};

struct TestNew6
{
  Matrix<float,2,2,DontAlign> m; // good: no alignment requested
  float f;
};

template<bool Align> struct Depends
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(Align)
  Vector2d m;
  float f;
};

template<typename T>
void check_unalignedassert_good()
{
  T *x, *y;
  x = new T;
  delete x;
  y = new T[2];
  delete[] y;
}

#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
template<typename T>
void construct_at_boundary(int boundary)
{
  char buf[sizeof(T)+256];
  size_t _buf = reinterpret_cast<internal::UIntPtr>(buf);
  _buf += (EIGEN_MAX_ALIGN_BYTES - (_buf % EIGEN_MAX_ALIGN_BYTES)); // make 16/32/...-byte aligned
  _buf += boundary; // make exact boundary-aligned
  T *x = ::new(reinterpret_cast<void*>(_buf)) T;
  x[0].setZero(); // just in order to silence warnings
  x->~T();
}
#endif

void unalignedassert()
{
#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
  construct_at_boundary<Vector2f>(4);
  construct_at_boundary<Vector3f>(4);
  construct_at_boundary<Vector4f>(16);
  construct_at_boundary<Vector6f>(4);
  construct_at_boundary<Vector8f>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Vector12f>(16);
  construct_at_boundary<Matrix2f>(16);
  construct_at_boundary<Matrix3f>(4);
  construct_at_boundary<Matrix4f>(EIGEN_MAX_ALIGN_BYTES);

  construct_at_boundary<Vector2d>(16);
  construct_at_boundary<Vector3d>(4);
  construct_at_boundary<Vector4d>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Vector5d>(4);
  construct_at_boundary<Vector6d>(16);
  construct_at_boundary<Vector7d>(4);
  construct_at_boundary<Vector8d>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Vector9d>(4);
  construct_at_boundary<Vector10d>(16);
  construct_at_boundary<Vector12d>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Matrix2d>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Matrix3d>(4);
  construct_at_boundary<Matrix4d>(EIGEN_MAX_ALIGN_BYTES);

  construct_at_boundary<Vector2cf>(16);
  construct_at_boundary<Vector3cf>(4);
  construct_at_boundary<Vector2cd>(EIGEN_MAX_ALIGN_BYTES);
  construct_at_boundary<Vector3cd>(16);
#endif

  check_unalignedassert_good<TestNew1>();
  check_unalignedassert_good<TestNew2>();
  check_unalignedassert_good<TestNew3>();

  check_unalignedassert_good<TestNew4>();
  check_unalignedassert_good<TestNew5>();
  check_unalignedassert_good<TestNew6>();
  check_unalignedassert_good<Depends<true> >();

#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
  if(EIGEN_MAX_ALIGN_BYTES>=16)
  {
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector4f>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector8f>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector12f>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector2d>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector4d>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector6d>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector8d>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector10d>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector12d>(8));
    // Complexes are disabled because the compiler might aggressively vectorize
    // the initialization of complex coeffs to 0 before we can check for alignedness
    //VERIFY_RAISES_ASSERT(construct_at_boundary<Vector2cf>(8));
    VERIFY_RAISES_ASSERT(construct_at_boundary<Vector4i>(8));
  }
  for(int b=8; b<EIGEN_MAX_ALIGN_BYTES; b+=8)
  {
    if(b<32)  VERIFY_RAISES_ASSERT(construct_at_boundary<Vector8f>(b));
    if(b<64)  VERIFY_RAISES_ASSERT(construct_at_boundary<Matrix4f>(b));
    if(b<32)  VERIFY_RAISES_ASSERT(construct_at_boundary<Vector4d>(b));
    if(b<32)  VERIFY_RAISES_ASSERT(construct_at_boundary<Matrix2d>(b));
    if(b<128) VERIFY_RAISES_ASSERT(construct_at_boundary<Matrix4d>(b));
    //if(b<32)  VERIFY_RAISES_ASSERT(construct_at_boundary<Vector2cd>(b));
  }
#endif
}

void test_unalignedassert()
{
  CALL_SUBTEST(unalignedassert());
}
