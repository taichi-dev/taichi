// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/SVD>

#define SVD_DEFAULT(M) JacobiSVD<M>
#define SVD_FOR_MIN_NORM(M) JacobiSVD<M,ColPivHouseholderQRPreconditioner>
#include "svd_common.h"

// Check all variants of JacobiSVD
template<typename MatrixType>
void jacobisvd(const MatrixType& a = MatrixType(), bool pickrandom = true)
{
  MatrixType m = a;
  if(pickrandom)
    svd_fill_random(m);

  CALL_SUBTEST(( svd_test_all_computation_options<JacobiSVD<MatrixType, FullPivHouseholderQRPreconditioner> >(m, true)  )); // check full only
  CALL_SUBTEST(( svd_test_all_computation_options<JacobiSVD<MatrixType, ColPivHouseholderQRPreconditioner>  >(m, false) ));
  CALL_SUBTEST(( svd_test_all_computation_options<JacobiSVD<MatrixType, HouseholderQRPreconditioner>        >(m, false) ));
  if(m.rows()==m.cols())
    CALL_SUBTEST(( svd_test_all_computation_options<JacobiSVD<MatrixType, NoQRPreconditioner>               >(m, false) ));
}

template<typename MatrixType> void jacobisvd_verify_assert(const MatrixType& m)
{
  svd_verify_assert<JacobiSVD<MatrixType> >(m);
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };


  MatrixType a = MatrixType::Zero(rows, cols);
  a.setZero();

  if (ColsAtCompileTime == Dynamic)
  {
    JacobiSVD<MatrixType, FullPivHouseholderQRPreconditioner> svd_fullqr;
    VERIFY_RAISES_ASSERT(svd_fullqr.compute(a, ComputeFullU|ComputeThinV))
    VERIFY_RAISES_ASSERT(svd_fullqr.compute(a, ComputeThinU|ComputeThinV))
    VERIFY_RAISES_ASSERT(svd_fullqr.compute(a, ComputeThinU|ComputeFullV))
  }
}

template<typename MatrixType>
void jacobisvd_method()
{
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.jacobiSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixV());
  VERIFY_IS_APPROX(m.jacobiSvd(ComputeFullU|ComputeFullV).solve(m), m);
}

namespace Foo {
// older compiler require a default constructor for Bar
// cf: https://stackoverflow.com/questions/7411515/
class Bar {public: Bar() {}};
bool operator<(const Bar&, const Bar&) { return true; }
}
// regression test for a very strange MSVC issue for which simply
// including SVDBase.h messes up with std::max and custom scalar type
void msvc_workaround()
{
  const Foo::Bar a;
  const Foo::Bar b;
  std::max EIGEN_NOT_A_MACRO (a,b);
}

void test_jacobisvd()
{
  CALL_SUBTEST_3(( jacobisvd_verify_assert(Matrix3f()) ));
  CALL_SUBTEST_4(( jacobisvd_verify_assert(Matrix4d()) ));
  CALL_SUBTEST_7(( jacobisvd_verify_assert(MatrixXf(10,12)) ));
  CALL_SUBTEST_8(( jacobisvd_verify_assert(MatrixXcd(7,5)) ));
  
  CALL_SUBTEST_11(svd_all_trivial_2x2(jacobisvd<Matrix2cd>));
  CALL_SUBTEST_12(svd_all_trivial_2x2(jacobisvd<Matrix2d>));

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_3(( jacobisvd<Matrix3f>() ));
    CALL_SUBTEST_4(( jacobisvd<Matrix4d>() ));
    CALL_SUBTEST_5(( jacobisvd<Matrix<float,3,5> >() ));
    CALL_SUBTEST_6(( jacobisvd<Matrix<double,Dynamic,2> >(Matrix<double,Dynamic,2>(10,2)) ));

    int r = internal::random<int>(1, 30),
        c = internal::random<int>(1, 30);
    
    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)
    
    CALL_SUBTEST_10(( jacobisvd<MatrixXd>(MatrixXd(r,c)) ));
    CALL_SUBTEST_7(( jacobisvd<MatrixXf>(MatrixXf(r,c)) ));
    CALL_SUBTEST_8(( jacobisvd<MatrixXcd>(MatrixXcd(r,c)) ));
    (void) r;
    (void) c;

    // Test on inf/nan matrix
    CALL_SUBTEST_7(  (svd_inf_nan<JacobiSVD<MatrixXf>, MatrixXf>()) );
    CALL_SUBTEST_10( (svd_inf_nan<JacobiSVD<MatrixXd>, MatrixXd>()) );

    // bug1395 test compile-time vectors as input
    CALL_SUBTEST_13(( jacobisvd_verify_assert(Matrix<double,6,1>()) ));
    CALL_SUBTEST_13(( jacobisvd_verify_assert(Matrix<double,1,6>()) ));
    CALL_SUBTEST_13(( jacobisvd_verify_assert(Matrix<double,Dynamic,1>(r)) ));
    CALL_SUBTEST_13(( jacobisvd_verify_assert(Matrix<double,1,Dynamic>(c)) ));
  }

  CALL_SUBTEST_7(( jacobisvd<MatrixXf>(MatrixXf(internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/2), internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/2))) ));
  CALL_SUBTEST_8(( jacobisvd<MatrixXcd>(MatrixXcd(internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/3), internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/3))) ));

  // test matrixbase method
  CALL_SUBTEST_1(( jacobisvd_method<Matrix2cd>() ));
  CALL_SUBTEST_3(( jacobisvd_method<Matrix3f>() ));

  // Test problem size constructors
  CALL_SUBTEST_7( JacobiSVD<MatrixXf>(10,10) );

  // Check that preallocation avoids subsequent mallocs
  CALL_SUBTEST_9( svd_preallocate<void>() );

  CALL_SUBTEST_2( svd_underoverflow<void>() );

  msvc_workaround();
}
