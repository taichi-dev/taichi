// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>

template<typename Scalar,int Size> void homogeneous(void)
{
  /* this test covers the following files:
     Homogeneous.h
  */

  typedef Matrix<Scalar,Size,Size> MatrixType;
  typedef Matrix<Scalar,Size,1, ColMajor> VectorType;

  typedef Matrix<Scalar,Size+1,Size> HMatrixType;
  typedef Matrix<Scalar,Size+1,1> HVectorType;

  typedef Matrix<Scalar,Size,Size+1>   T1MatrixType;
  typedef Matrix<Scalar,Size+1,Size+1> T2MatrixType;
  typedef Matrix<Scalar,Size+1,Size> T3MatrixType;

  VectorType v0 = VectorType::Random(),
             ones = VectorType::Ones();

  HVectorType hv0 = HVectorType::Random();

  MatrixType m0 = MatrixType::Random();

  HMatrixType hm0 = HMatrixType::Random();

  hv0 << v0, 1;
  VERIFY_IS_APPROX(v0.homogeneous(), hv0);
  VERIFY_IS_APPROX(v0, hv0.hnormalized());
  
  VERIFY_IS_APPROX(v0.homogeneous().sum(), hv0.sum());
  VERIFY_IS_APPROX(v0.homogeneous().minCoeff(), hv0.minCoeff());
  VERIFY_IS_APPROX(v0.homogeneous().maxCoeff(), hv0.maxCoeff());

  hm0 << m0, ones.transpose();
  VERIFY_IS_APPROX(m0.colwise().homogeneous(), hm0);
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());
  hm0.row(Size-1).setRandom();
  for(int j=0; j<Size; ++j)
    m0.col(j) = hm0.col(j).head(Size) / hm0(Size,j);
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());

  T1MatrixType t1 = T1MatrixType::Random();
  VERIFY_IS_APPROX(t1 * (v0.homogeneous().eval()), t1 * v0.homogeneous());
  VERIFY_IS_APPROX(t1 * (m0.colwise().homogeneous().eval()), t1 * m0.colwise().homogeneous());

  T2MatrixType t2 = T2MatrixType::Random();
  VERIFY_IS_APPROX(t2 * (v0.homogeneous().eval()), t2 * v0.homogeneous());
  VERIFY_IS_APPROX(t2 * (m0.colwise().homogeneous().eval()), t2 * m0.colwise().homogeneous());
  VERIFY_IS_APPROX(t2 * (v0.homogeneous().asDiagonal()), t2 * hv0.asDiagonal());
  VERIFY_IS_APPROX((v0.homogeneous().asDiagonal()) * t2, hv0.asDiagonal() * t2);

  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t2,
                    v0.transpose().rowwise().homogeneous() * t2);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t2,
                    m0.transpose().rowwise().homogeneous() * t2);

  T3MatrixType t3 = T3MatrixType::Random();
  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t3,
                    v0.transpose().rowwise().homogeneous() * t3);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t3,
                    m0.transpose().rowwise().homogeneous() * t3);

  // test product with a Transform object
  Transform<Scalar, Size, Affine> aff;
  Transform<Scalar, Size, AffineCompact> caff;
  Transform<Scalar, Size, Projective> proj;
  Matrix<Scalar, Size, Dynamic>   pts;
  Matrix<Scalar, Size+1, Dynamic> pts1, pts2;

  aff.affine().setRandom();
  proj = caff = aff;
  pts.setRandom(Size,internal::random<int>(1,20));
  
  pts1 = pts.colwise().homogeneous();
  VERIFY_IS_APPROX(aff  * pts.colwise().homogeneous(), (aff  * pts1).colwise().hnormalized());
  VERIFY_IS_APPROX(caff * pts.colwise().homogeneous(), (caff * pts1).colwise().hnormalized());
  VERIFY_IS_APPROX(proj * pts.colwise().homogeneous(), (proj * pts1));

  VERIFY_IS_APPROX((aff  * pts1).colwise().hnormalized(),  aff  * pts);
  VERIFY_IS_APPROX((caff * pts1).colwise().hnormalized(), caff * pts);
  
  pts2 = pts1;
  pts2.row(Size).setRandom();
  VERIFY_IS_APPROX((aff  * pts2).colwise().hnormalized(), aff  * pts2.colwise().hnormalized());
  VERIFY_IS_APPROX((caff * pts2).colwise().hnormalized(), caff * pts2.colwise().hnormalized());
  VERIFY_IS_APPROX((proj * pts2).colwise().hnormalized(), (proj * pts2.colwise().hnormalized().colwise().homogeneous()).colwise().hnormalized());
  
  // Test combination of homogeneous
  
  VERIFY_IS_APPROX( (t2 * v0.homogeneous()).hnormalized(),
                       (t2.template topLeftCorner<Size,Size>() * v0 + t2.template topRightCorner<Size,1>())
                     / ((t2.template bottomLeftCorner<1,Size>()*v0).value() + t2(Size,Size)) );
  
  VERIFY_IS_APPROX( (t2 * pts.colwise().homogeneous()).colwise().hnormalized(),
                    (Matrix<Scalar, Size+1, Dynamic>(t2 * pts1).colwise().hnormalized()) );
  
  VERIFY_IS_APPROX( (t2 .lazyProduct( v0.homogeneous() )).hnormalized(), (t2 * v0.homogeneous()).hnormalized() );
  VERIFY_IS_APPROX( (t2 .lazyProduct  ( pts.colwise().homogeneous() )).colwise().hnormalized(), (t2 * pts1).colwise().hnormalized() );
  
  VERIFY_IS_APPROX( (v0.transpose().homogeneous() .lazyProduct( t2 )).hnormalized(), (v0.transpose().homogeneous()*t2).hnormalized() );
  VERIFY_IS_APPROX( (pts.transpose().rowwise().homogeneous() .lazyProduct( t2 )).rowwise().hnormalized(), (pts1.transpose()*t2).rowwise().hnormalized() );

  VERIFY_IS_APPROX( (t2.template triangularView<Lower>() * v0.homogeneous()).eval(), (t2.template triangularView<Lower>()*hv0) );
}

void test_geo_homogeneous()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( homogeneous<float,1>() ));
    CALL_SUBTEST_2(( homogeneous<double,3>() ));
    CALL_SUBTEST_3(( homogeneous<double,8>() ));
  }
}
