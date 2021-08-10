// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"

#include <Eigen/Core>

using internal::UIntPtr;

#if EIGEN_HAS_RVALUE_REFERENCES
template <typename MatrixType>
void rvalue_copyassign(const MatrixType& m)
{

  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  
  // create a temporary which we are about to destroy by moving
  MatrixType tmp = m;
  UIntPtr src_address = reinterpret_cast<UIntPtr>(tmp.data());
  
  Eigen::internal::set_is_malloc_allowed(false); // moving from an rvalue reference shall never allocate
  // move the temporary to n
  MatrixType n = std::move(tmp);
  UIntPtr dst_address = reinterpret_cast<UIntPtr>(n.data());
  if (MatrixType::RowsAtCompileTime==Dynamic|| MatrixType::ColsAtCompileTime==Dynamic)
  {
    // verify that we actually moved the guts
    VERIFY_IS_EQUAL(src_address, dst_address);
    VERIFY_IS_EQUAL(tmp.size(), 0);
    VERIFY_IS_EQUAL(reinterpret_cast<UIntPtr>(tmp.data()), UIntPtr(0));
  }

  // verify that the content did not change
  Scalar abs_diff = (m-n).array().abs().sum();
  VERIFY_IS_EQUAL(abs_diff, Scalar(0));
  Eigen::internal::set_is_malloc_allowed(true);
}
template<typename TranspositionsType>
void rvalue_transpositions(Index rows)
{
  typedef typename TranspositionsType::IndicesType PermutationVectorType;

  PermutationVectorType vec;
  randomPermutationVector(vec, rows);
  TranspositionsType t0(vec);

  Eigen::internal::set_is_malloc_allowed(false); // moving from an rvalue reference shall never allocate

  UIntPtr t0_address = reinterpret_cast<UIntPtr>(t0.indices().data());

  // Move constructors:
  TranspositionsType t1 = std::move(t0);
  UIntPtr t1_address = reinterpret_cast<UIntPtr>(t1.indices().data());
  VERIFY_IS_EQUAL(t0_address, t1_address);
  // t0 must be de-allocated:
  VERIFY_IS_EQUAL(t0.size(), 0);
  VERIFY_IS_EQUAL(reinterpret_cast<UIntPtr>(t0.indices().data()), UIntPtr(0));


  // Move assignment:
  t0 = std::move(t1);
  t0_address = reinterpret_cast<UIntPtr>(t0.indices().data());
  VERIFY_IS_EQUAL(t0_address, t1_address);
  // t1 must be de-allocated:
  VERIFY_IS_EQUAL(t1.size(), 0);
  VERIFY_IS_EQUAL(reinterpret_cast<UIntPtr>(t1.indices().data()), UIntPtr(0));

  Eigen::internal::set_is_malloc_allowed(true);
}
#else
template <typename MatrixType>
void rvalue_copyassign(const MatrixType&) {}
template<typename TranspositionsType>
void rvalue_transpositions(Index) {}
#endif

void test_rvalue_types()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(rvalue_copyassign( MatrixXf::Random(50,50).eval() ));
    CALL_SUBTEST_1(rvalue_copyassign( ArrayXXf::Random(50,50).eval() ));

    CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,1,Dynamic>::Random(50).eval() ));
    CALL_SUBTEST_1(rvalue_copyassign( Array<float,1,Dynamic>::Random(50).eval() ));

    CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,Dynamic,1>::Random(50).eval() ));
    CALL_SUBTEST_1(rvalue_copyassign( Array<float,Dynamic,1>::Random(50).eval() ));

    CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,1>::Random().eval() ));
    CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,1>::Random().eval() ));
    CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,1>::Random().eval() ));

    CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,2>::Random().eval() ));
    CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,3>::Random().eval() ));
    CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,4>::Random().eval() ));
  
    CALL_SUBTEST_3((rvalue_transpositions<PermutationMatrix<Dynamic, Dynamic, int> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((rvalue_transpositions<PermutationMatrix<Dynamic, Dynamic, Index> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_4((rvalue_transpositions<Transpositions<Dynamic, Dynamic, int> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_4((rvalue_transpositions<Transpositions<Dynamic, Dynamic, Index> >(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
  }
}
