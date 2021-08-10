// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

/* this test covers the following files:
   Geometry/OrthoMethods.h
*/

template<typename Scalar> void orthomethods_3()
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;

  typedef Matrix<Scalar,4,1> Vector4;

  Vector3 v0 = Vector3::Random(),
          v1 = Vector3::Random(),
          v2 = Vector3::Random();

  // cross product
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(v2).dot(v1), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(v1.dot(v1.cross(v2)), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(v2).dot(v2), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(v2.dot(v1.cross(v2)), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(Vector3::Random()).dot(v1), Scalar(1));
  Matrix3 mat3;
  mat3 << v0.normalized(),
         (v0.cross(v1)).normalized(),
         (v0.cross(v1).cross(v0)).normalized();
  VERIFY(mat3.isUnitary());
  
  mat3.setRandom();
  VERIFY_IS_APPROX(v0.cross(mat3*v1), -(mat3*v1).cross(v0));
  VERIFY_IS_APPROX(v0.cross(mat3.lazyProduct(v1)), -(mat3.lazyProduct(v1)).cross(v0));

  // colwise/rowwise cross product
  mat3.setRandom();
  Vector3 vec3 = Vector3::Random();
  Matrix3 mcross;
  int i = internal::random<int>(0,2);
  mcross = mat3.colwise().cross(vec3);
  VERIFY_IS_APPROX(mcross.col(i), mat3.col(i).cross(vec3));
  
  VERIFY_IS_MUCH_SMALLER_THAN((mat3.adjoint() * mat3.colwise().cross(vec3)).diagonal().cwiseAbs().sum(), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN((mat3.adjoint() * mat3.colwise().cross(Vector3::Random())).diagonal().cwiseAbs().sum(), Scalar(1));
  
  VERIFY_IS_MUCH_SMALLER_THAN((vec3.adjoint() * mat3.colwise().cross(vec3)).cwiseAbs().sum(), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN((vec3.adjoint() * Matrix3::Random().colwise().cross(vec3)).cwiseAbs().sum(), Scalar(1));
  
  mcross = mat3.rowwise().cross(vec3);
  VERIFY_IS_APPROX(mcross.row(i), mat3.row(i).cross(vec3));

  // cross3
  Vector4 v40 = Vector4::Random(),
          v41 = Vector4::Random(),
          v42 = Vector4::Random();
  v40.w() = v41.w() = v42.w() = 0;
  v42.template head<3>() = v40.template head<3>().cross(v41.template head<3>());
  VERIFY_IS_APPROX(v40.cross3(v41), v42);
  VERIFY_IS_MUCH_SMALLER_THAN(v40.cross3(Vector4::Random()).dot(v40), Scalar(1));
  
  // check mixed product
  typedef Matrix<RealScalar, 3, 1> RealVector3;
  RealVector3 rv1 = RealVector3::Random();
  VERIFY_IS_APPROX(v1.cross(rv1.template cast<Scalar>()), v1.cross(rv1));
  VERIFY_IS_APPROX(rv1.template cast<Scalar>().cross(v1), rv1.cross(v1));
}

template<typename Scalar, int Size> void orthomethods(int size=Size)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,Size,1> VectorType;
  typedef Matrix<Scalar,3,Size> Matrix3N;
  typedef Matrix<Scalar,Size,3> MatrixN3;
  typedef Matrix<Scalar,3,1> Vector3;

  VectorType v0 = VectorType::Random(size);

  // unitOrthogonal
  VERIFY_IS_MUCH_SMALLER_THAN(v0.unitOrthogonal().dot(v0), Scalar(1));
  VERIFY_IS_APPROX(v0.unitOrthogonal().norm(), RealScalar(1));

  if (size>=3)
  {
    v0.template head<2>().setZero();
    v0.tail(size-2).setRandom();

    VERIFY_IS_MUCH_SMALLER_THAN(v0.unitOrthogonal().dot(v0), Scalar(1));
    VERIFY_IS_APPROX(v0.unitOrthogonal().norm(), RealScalar(1));
  }

  // colwise/rowwise cross product
  Vector3 vec3 = Vector3::Random();
  int i = internal::random<int>(0,size-1);

  Matrix3N mat3N(3,size), mcross3N(3,size);
  mat3N.setRandom();
  mcross3N = mat3N.colwise().cross(vec3);
  VERIFY_IS_APPROX(mcross3N.col(i), mat3N.col(i).cross(vec3));

  MatrixN3 matN3(size,3), mcrossN3(size,3);
  matN3.setRandom();
  mcrossN3 = matN3.rowwise().cross(vec3);
  VERIFY_IS_APPROX(mcrossN3.row(i), matN3.row(i).cross(vec3));
}

void test_geo_orthomethods()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( orthomethods_3<float>() );
    CALL_SUBTEST_2( orthomethods_3<double>() );
    CALL_SUBTEST_4( orthomethods_3<std::complex<double> >() );
    CALL_SUBTEST_1( (orthomethods<float,2>()) );
    CALL_SUBTEST_2( (orthomethods<double,2>()) );
    CALL_SUBTEST_1( (orthomethods<float,3>()) );
    CALL_SUBTEST_2( (orthomethods<double,3>()) );
    CALL_SUBTEST_3( (orthomethods<float,7>()) );
    CALL_SUBTEST_4( (orthomethods<std::complex<double>,8>()) );
    CALL_SUBTEST_5( (orthomethods<float,Dynamic>(36)) );
    CALL_SUBTEST_6( (orthomethods<double,Dynamic>(35)) );
  }
}
