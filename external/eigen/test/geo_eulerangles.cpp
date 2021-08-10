// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>


template<typename Scalar>
void verify_euler(const Matrix<Scalar,3,1>& ea, int i, int j, int k)
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef AngleAxis<Scalar> AngleAxisx;
  using std::abs;
  Matrix3 m(AngleAxisx(ea[0], Vector3::Unit(i)) * AngleAxisx(ea[1], Vector3::Unit(j)) * AngleAxisx(ea[2], Vector3::Unit(k)));
  Vector3 eabis = m.eulerAngles(i, j, k);
  Matrix3 mbis(AngleAxisx(eabis[0], Vector3::Unit(i)) * AngleAxisx(eabis[1], Vector3::Unit(j)) * AngleAxisx(eabis[2], Vector3::Unit(k))); 
  VERIFY_IS_APPROX(m,  mbis); 
  /* If I==K, and ea[1]==0, then there no unique solution. */ 
  /* The remark apply in the case where I!=K, and |ea[1]| is close to pi/2. */ 
  if( (i!=k || ea[1]!=0) && (i==k || !internal::isApprox(abs(ea[1]),Scalar(EIGEN_PI/2),test_precision<Scalar>())) ) 
    VERIFY((ea-eabis).norm() <= test_precision<Scalar>());
  
  // approx_or_less_than does not work for 0
  VERIFY(0 < eabis[0] || test_isMuchSmallerThan(eabis[0], Scalar(1)));
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[0], Scalar(EIGEN_PI));
  VERIFY_IS_APPROX_OR_LESS_THAN(-Scalar(EIGEN_PI), eabis[1]);
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[1], Scalar(EIGEN_PI));
  VERIFY_IS_APPROX_OR_LESS_THAN(-Scalar(EIGEN_PI), eabis[2]);
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[2], Scalar(EIGEN_PI));
}

template<typename Scalar> void check_all_var(const Matrix<Scalar,3,1>& ea)
{
  verify_euler(ea, 0,1,2);
  verify_euler(ea, 0,1,0);
  verify_euler(ea, 0,2,1);
  verify_euler(ea, 0,2,0);

  verify_euler(ea, 1,2,0);
  verify_euler(ea, 1,2,1);
  verify_euler(ea, 1,0,2);
  verify_euler(ea, 1,0,1);

  verify_euler(ea, 2,0,1);
  verify_euler(ea, 2,0,2);
  verify_euler(ea, 2,1,0);
  verify_euler(ea, 2,1,2);
}

template<typename Scalar> void eulerangles()
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Array<Scalar,3,1> Array3;
  typedef Quaternion<Scalar> Quaternionx;
  typedef AngleAxis<Scalar> AngleAxisx;

  Scalar a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));
  Quaternionx q1;
  q1 = AngleAxisx(a, Vector3::Random().normalized());
  Matrix3 m;
  m = q1;
  
  Vector3 ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with purely random Quaternion:
  q1.coeffs() = Quaternionx::Coefficients::Random().normalized();
  m = q1;
  ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with random angles in range [0:pi]x[-pi:pi]x[-pi:pi].
  ea = (Array3::Random() + Array3(1,0,0))*Scalar(EIGEN_PI)*Array3(0.5,1,1);
  check_all_var(ea);
  
  ea[2] = ea[0] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[0] = ea[1] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[1] = 0;
  check_all_var(ea);
  
  ea.head(2).setZero();
  check_all_var(ea);
  
  ea.setZero();
  check_all_var(ea);
}

void test_geo_eulerangles()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( eulerangles<float>() );
    CALL_SUBTEST_2( eulerangles<double>() );
  }
}
