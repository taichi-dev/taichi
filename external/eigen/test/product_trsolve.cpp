// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#define VERIFY_TRSM(TRI,XB) { \
    (XB).setRandom(); ref = (XB); \
    (TRI).solveInPlace(XB); \
    VERIFY_IS_APPROX((TRI).toDenseMatrix() * (XB), ref); \
    (XB).setRandom(); ref = (XB); \
    (XB) = (TRI).solve(XB); \
    VERIFY_IS_APPROX((TRI).toDenseMatrix() * (XB), ref); \
  }

#define VERIFY_TRSM_ONTHERIGHT(TRI,XB) { \
    (XB).setRandom(); ref = (XB); \
    (TRI).transpose().template solveInPlace<OnTheRight>(XB.transpose()); \
    VERIFY_IS_APPROX((XB).transpose() * (TRI).transpose().toDenseMatrix(), ref.transpose()); \
    (XB).setRandom(); ref = (XB); \
    (XB).transpose() = (TRI).transpose().template solve<OnTheRight>(XB.transpose()); \
    VERIFY_IS_APPROX((XB).transpose() * (TRI).transpose().toDenseMatrix(), ref.transpose()); \
  }

template<typename Scalar,int Size, int Cols> void trsolve(int size=Size,int cols=Cols)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Matrix<Scalar,Size,Size,ColMajor> cmLhs(size,size);
  Matrix<Scalar,Size,Size,RowMajor> rmLhs(size,size);

  enum {  colmajor = Size==1 ? RowMajor : ColMajor,
          rowmajor = Cols==1 ? ColMajor : RowMajor };
  Matrix<Scalar,Size,Cols,colmajor> cmRhs(size,cols);
  Matrix<Scalar,Size,Cols,rowmajor> rmRhs(size,cols);
  Matrix<Scalar,Dynamic,Dynamic,colmajor> ref(size,cols);

  cmLhs.setRandom(); cmLhs *= static_cast<RealScalar>(0.1); cmLhs.diagonal().array() += static_cast<RealScalar>(1);
  rmLhs.setRandom(); rmLhs *= static_cast<RealScalar>(0.1); rmLhs.diagonal().array() += static_cast<RealScalar>(1);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(cmLhs.adjoint()  .template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);
  VERIFY_TRSM(cmLhs.adjoint()  .template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM(rmLhs            .template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);


  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(rmLhs            .template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);

  int c = internal::random<int>(0,cols-1);
  VERIFY_TRSM(rmLhs.template triangularView<Lower>(), rmRhs.col(c));
  VERIFY_TRSM(cmLhs.template triangularView<Lower>(), rmRhs.col(c));

  // destination with a non-default inner-stride
  // see bug 1741
  {
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;
    MatrixX buffer(2*cmRhs.rows(),2*cmRhs.cols());
    Map<Matrix<Scalar,Size,Cols,colmajor>,0,Stride<Dynamic,2> > map1(buffer.data(),cmRhs.rows(),cmRhs.cols(),Stride<Dynamic,2>(2*cmRhs.outerStride(),2));
    Map<Matrix<Scalar,Size,Cols,rowmajor>,0,Stride<Dynamic,2> > map2(buffer.data(),rmRhs.rows(),rmRhs.cols(),Stride<Dynamic,2>(2*rmRhs.outerStride(),2));
    buffer.setZero();
    VERIFY_TRSM(cmLhs.conjugate().template triangularView<Lower>(), map1);
    buffer.setZero();
    VERIFY_TRSM(cmLhs            .template triangularView<Lower>(), map2);
  }

  if(Size==Dynamic)
  {
    cmLhs.resize(0,0);
    cmRhs.resize(0,cmRhs.cols());
    Matrix<Scalar,Size,Cols,colmajor> res = cmLhs.template triangularView<Lower>().solve(cmRhs);
    VERIFY_IS_EQUAL(res.rows(),0);
    VERIFY_IS_EQUAL(res.cols(),cmRhs.cols());
    res = cmRhs;
    cmLhs.template triangularView<Lower>().solveInPlace(res);
    VERIFY_IS_EQUAL(res.rows(),0);
    VERIFY_IS_EQUAL(res.cols(),cmRhs.cols());
  }
}

void test_product_trsolve()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    // matrices
    CALL_SUBTEST_1((trsolve<float,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2((trsolve<double,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_3((trsolve<std::complex<float>,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2),internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))));
    CALL_SUBTEST_4((trsolve<std::complex<double>,Dynamic,Dynamic>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2),internal::random<int>(1,EIGEN_TEST_MAX_SIZE/2))));

    // vectors
    CALL_SUBTEST_5((trsolve<float,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_6((trsolve<double,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_7((trsolve<std::complex<float>,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_8((trsolve<std::complex<double>,Dynamic,1>(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))));
    
    // meta-unrollers
    CALL_SUBTEST_9((trsolve<float,4,1>()));
    CALL_SUBTEST_10((trsolve<double,4,1>()));
    CALL_SUBTEST_11((trsolve<std::complex<float>,4,1>()));
    CALL_SUBTEST_12((trsolve<float,1,1>()));
    CALL_SUBTEST_13((trsolve<float,1,2>()));
    CALL_SUBTEST_14((trsolve<float,3,1>()));
    
  }
}
