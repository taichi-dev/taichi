// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Mathieu Gautier <mathieu.gautier@cea.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

template<typename T> T bounded_acos(T v)
{
  using std::acos;
  using std::min;
  using std::max;
  return acos((max)(T(-1),(min)(v,T(1))));
}

template<typename QuatType> void check_slerp(const QuatType& q0, const QuatType& q1)
{
  using std::abs;
  typedef typename QuatType::Scalar Scalar;
  typedef AngleAxis<Scalar> AA;

  Scalar largeEps = test_precision<Scalar>();

  Scalar theta_tot = AA(q1*q0.inverse()).angle();
  if(theta_tot>Scalar(EIGEN_PI))
    theta_tot = Scalar(2.)*Scalar(EIGEN_PI)-theta_tot;
  for(Scalar t=0; t<=Scalar(1.001); t+=Scalar(0.1))
  {
    QuatType q = q0.slerp(t,q1);
    Scalar theta = AA(q*q0.inverse()).angle();
    VERIFY(abs(q.norm() - 1) < largeEps);
    if(theta_tot==0)  VERIFY(theta_tot==0);
    else              VERIFY(abs(theta - t * theta_tot) < largeEps);
  }
}

template<typename Scalar, int Options> void quaternion(void)
{
  /* this test covers the following files:
     Quaternion.h
  */
  using std::abs;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Quaternion<Scalar,Options> Quaternionx;
  typedef AngleAxis<Scalar> AngleAxisx;

  Scalar largeEps = test_precision<Scalar>();
  if (internal::is_same<Scalar,float>::value)
    largeEps = Scalar(1e-3);

  Scalar eps = internal::random<Scalar>() * Scalar(1e-2);

  Vector3 v0 = Vector3::Random(),
          v1 = Vector3::Random(),
          v2 = Vector3::Random(),
          v3 = Vector3::Random();

  Scalar  a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI)),
          b = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));

  // Quaternion: Identity(), setIdentity();
  Quaternionx q1, q2;
  q2.setIdentity();
  VERIFY_IS_APPROX(Quaternionx(Quaternionx::Identity()).coeffs(), q2.coeffs());
  q1.coeffs().setRandom();
  VERIFY_IS_APPROX(q1.coeffs(), (q1*q2).coeffs());

  // concatenation
  q1 *= q2;

  q1 = AngleAxisx(a, v0.normalized());
  q2 = AngleAxisx(a, v1.normalized());

  // angular distance
  Scalar refangle = abs(AngleAxisx(q1.inverse()*q2).angle());
  if (refangle>Scalar(EIGEN_PI))
    refangle = Scalar(2)*Scalar(EIGEN_PI) - refangle;

  if((q1.coeffs()-q2.coeffs()).norm() > 10*largeEps)
  {
    VERIFY_IS_MUCH_SMALLER_THAN(abs(q1.angularDistance(q2) - refangle), Scalar(1));
  }

  // rotation matrix conversion
  VERIFY_IS_APPROX(q1 * v2, q1.toRotationMatrix() * v2);
  VERIFY_IS_APPROX(q1 * q2 * v2,
    q1.toRotationMatrix() * q2.toRotationMatrix() * v2);

  VERIFY(  (q2*q1).isApprox(q1*q2, largeEps)
        || !(q2 * q1 * v2).isApprox(q1.toRotationMatrix() * q2.toRotationMatrix() * v2));

  q2 = q1.toRotationMatrix();
  VERIFY_IS_APPROX(q1*v1,q2*v1);

  Matrix3 rot1(q1);
  VERIFY_IS_APPROX(q1*v1,rot1*v1);
  Quaternionx q3(rot1.transpose()*rot1);
  VERIFY_IS_APPROX(q3*v1,v1);


  // angle-axis conversion
  AngleAxisx aa = AngleAxisx(q1);
  VERIFY_IS_APPROX(q1 * v1, Quaternionx(aa) * v1);

  // Do not execute the test if the rotation angle is almost zero, or
  // the rotation axis and v1 are almost parallel.
  if (abs(aa.angle()) > 5*test_precision<Scalar>()
      && (aa.axis() - v1.normalized()).norm() < Scalar(1.99)
      && (aa.axis() + v1.normalized()).norm() < Scalar(1.99))
  {
    VERIFY_IS_NOT_APPROX(q1 * v1, Quaternionx(AngleAxisx(aa.angle()*2,aa.axis())) * v1);
  }

  // from two vector creation
  VERIFY_IS_APPROX( v2.normalized(),(q2.setFromTwoVectors(v1, v2)*v1).normalized());
  VERIFY_IS_APPROX( v1.normalized(),(q2.setFromTwoVectors(v1, v1)*v1).normalized());
  VERIFY_IS_APPROX(-v1.normalized(),(q2.setFromTwoVectors(v1,-v1)*v1).normalized());
  if (internal::is_same<Scalar,double>::value)
  {
    v3 = (v1.array()+eps).matrix();
    VERIFY_IS_APPROX( v3.normalized(),(q2.setFromTwoVectors(v1, v3)*v1).normalized());
    VERIFY_IS_APPROX(-v3.normalized(),(q2.setFromTwoVectors(v1,-v3)*v1).normalized());
  }

  // from two vector creation static function
  VERIFY_IS_APPROX( v2.normalized(),(Quaternionx::FromTwoVectors(v1, v2)*v1).normalized());
  VERIFY_IS_APPROX( v1.normalized(),(Quaternionx::FromTwoVectors(v1, v1)*v1).normalized());
  VERIFY_IS_APPROX(-v1.normalized(),(Quaternionx::FromTwoVectors(v1,-v1)*v1).normalized());
  if (internal::is_same<Scalar,double>::value)
  {
    v3 = (v1.array()+eps).matrix();
    VERIFY_IS_APPROX( v3.normalized(),(Quaternionx::FromTwoVectors(v1, v3)*v1).normalized());
    VERIFY_IS_APPROX(-v3.normalized(),(Quaternionx::FromTwoVectors(v1,-v3)*v1).normalized());
  }

  // inverse and conjugate
  VERIFY_IS_APPROX(q1 * (q1.inverse() * v1), v1);
  VERIFY_IS_APPROX(q1 * (q1.conjugate() * v1), v1);

  // test casting
  Quaternion<float> q1f = q1.template cast<float>();
  VERIFY_IS_APPROX(q1f.template cast<Scalar>(),q1);
  Quaternion<double> q1d = q1.template cast<double>();
  VERIFY_IS_APPROX(q1d.template cast<Scalar>(),q1);

  // test bug 369 - improper alignment.
  Quaternionx *q = new Quaternionx;
  delete q;

  q1 = Quaternionx::UnitRandom();
  q2 = Quaternionx::UnitRandom();
  check_slerp(q1,q2);

  q1 = AngleAxisx(b, v1.normalized());
  q2 = AngleAxisx(b+Scalar(EIGEN_PI), v1.normalized());
  check_slerp(q1,q2);

  q1 = AngleAxisx(b,  v1.normalized());
  q2 = AngleAxisx(-b, -v1.normalized());
  check_slerp(q1,q2);

  q1 = Quaternionx::UnitRandom();
  q2.coeffs() = -q1.coeffs();
  check_slerp(q1,q2);
}

template<typename Scalar> void mapQuaternion(void){
  typedef Map<Quaternion<Scalar>, Aligned> MQuaternionA;
  typedef Map<const Quaternion<Scalar>, Aligned> MCQuaternionA;
  typedef Map<Quaternion<Scalar> > MQuaternionUA;
  typedef Map<const Quaternion<Scalar> > MCQuaternionUA;
  typedef Quaternion<Scalar> Quaternionx;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef AngleAxis<Scalar> AngleAxisx;
  
  Vector3 v0 = Vector3::Random(),
          v1 = Vector3::Random();
  Scalar  a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));

  EIGEN_ALIGN_MAX Scalar array1[4];
  EIGEN_ALIGN_MAX Scalar array2[4];
  EIGEN_ALIGN_MAX Scalar array3[4+1];
  Scalar* array3unaligned = array3+1;
  
  MQuaternionA    mq1(array1);
  MCQuaternionA   mcq1(array1);
  MQuaternionA    mq2(array2);
  MQuaternionUA   mq3(array3unaligned);
  MCQuaternionUA  mcq3(array3unaligned);

//  std::cerr << array1 << " " << array2 << " " << array3 << "\n";
  mq1 = AngleAxisx(a, v0.normalized());
  mq2 = mq1;
  mq3 = mq1;

  Quaternionx q1 = mq1;
  Quaternionx q2 = mq2;
  Quaternionx q3 = mq3;
  Quaternionx q4 = MCQuaternionUA(array3unaligned);

  VERIFY_IS_APPROX(q1.coeffs(), q2.coeffs());
  VERIFY_IS_APPROX(q1.coeffs(), q3.coeffs());
  VERIFY_IS_APPROX(q4.coeffs(), q3.coeffs());
  #ifdef EIGEN_VECTORIZE
  if(internal::packet_traits<Scalar>::Vectorizable)
    VERIFY_RAISES_ASSERT((MQuaternionA(array3unaligned)));
  #endif
    
  VERIFY_IS_APPROX(mq1 * (mq1.inverse() * v1), v1);
  VERIFY_IS_APPROX(mq1 * (mq1.conjugate() * v1), v1);
  
  VERIFY_IS_APPROX(mcq1 * (mcq1.inverse() * v1), v1);
  VERIFY_IS_APPROX(mcq1 * (mcq1.conjugate() * v1), v1);
  
  VERIFY_IS_APPROX(mq3 * (mq3.inverse() * v1), v1);
  VERIFY_IS_APPROX(mq3 * (mq3.conjugate() * v1), v1);
  
  VERIFY_IS_APPROX(mcq3 * (mcq3.inverse() * v1), v1);
  VERIFY_IS_APPROX(mcq3 * (mcq3.conjugate() * v1), v1);
  
  VERIFY_IS_APPROX(mq1*mq2, q1*q2);
  VERIFY_IS_APPROX(mq3*mq2, q3*q2);
  VERIFY_IS_APPROX(mcq1*mq2, q1*q2);
  VERIFY_IS_APPROX(mcq3*mq2, q3*q2);

  // Bug 1461, compilation issue with Map<const Quat>::w(), and other reference/constness checks:
  VERIFY_IS_APPROX(mcq3.coeffs().x() + mcq3.coeffs().y() + mcq3.coeffs().z() + mcq3.coeffs().w(), mcq3.coeffs().sum());
  VERIFY_IS_APPROX(mcq3.x() + mcq3.y() + mcq3.z() + mcq3.w(), mcq3.coeffs().sum());
  mq3.w() = 1;
  const Quaternionx& cq3(q3);
  VERIFY( &cq3.x() == &q3.x() );
  const MQuaternionUA& cmq3(mq3);
  VERIFY( &cmq3.x() == &mq3.x() );
  // FIXME the following should be ok. The problem is that currently the LValueBit flag
  // is used to determine wether we can return a coeff by reference or not, which is not enough for Map<const ...>.
  //const MCQuaternionUA& cmcq3(mcq3);
  //VERIFY( &cmcq3.x() == &mcq3.x() );

  // test cast
  {
    Quaternion<float> q1f = mq1.template cast<float>();
    VERIFY_IS_APPROX(q1f.template cast<Scalar>(),mq1);
    Quaternion<double> q1d = mq1.template cast<double>();
    VERIFY_IS_APPROX(q1d.template cast<Scalar>(),mq1);
  }
}

template<typename Scalar> void quaternionAlignment(void){
  typedef Quaternion<Scalar,AutoAlign> QuaternionA;
  typedef Quaternion<Scalar,DontAlign> QuaternionUA;

  EIGEN_ALIGN_MAX Scalar array1[4];
  EIGEN_ALIGN_MAX Scalar array2[4];
  EIGEN_ALIGN_MAX Scalar array3[4+1];
  Scalar* arrayunaligned = array3+1;

  QuaternionA *q1 = ::new(reinterpret_cast<void*>(array1)) QuaternionA;
  QuaternionUA *q2 = ::new(reinterpret_cast<void*>(array2)) QuaternionUA;
  QuaternionUA *q3 = ::new(reinterpret_cast<void*>(arrayunaligned)) QuaternionUA;

  q1->coeffs().setRandom();
  *q2 = *q1;
  *q3 = *q1;

  VERIFY_IS_APPROX(q1->coeffs(), q2->coeffs());
  VERIFY_IS_APPROX(q1->coeffs(), q3->coeffs());
  #if defined(EIGEN_VECTORIZE) && EIGEN_MAX_STATIC_ALIGN_BYTES>0
  if(internal::packet_traits<Scalar>::Vectorizable && internal::packet_traits<Scalar>::size<=4)
    VERIFY_RAISES_ASSERT((::new(reinterpret_cast<void*>(arrayunaligned)) QuaternionA));
  #endif
}

template<typename PlainObjectType> void check_const_correctness(const PlainObjectType&)
{
  // there's a lot that we can't test here while still having this test compile!
  // the only possible approach would be to run a script trying to compile stuff and checking that it fails.
  // CMake can help with that.

  // verify that map-to-const don't have LvalueBit
  typedef typename internal::add_const<PlainObjectType>::type ConstPlainObjectType;
  VERIFY( !(internal::traits<Map<ConstPlainObjectType> >::Flags & LvalueBit) );
  VERIFY( !(internal::traits<Map<ConstPlainObjectType, Aligned> >::Flags & LvalueBit) );
  VERIFY( !(Map<ConstPlainObjectType>::Flags & LvalueBit) );
  VERIFY( !(Map<ConstPlainObjectType, Aligned>::Flags & LvalueBit) );
}

void test_geo_quaternion()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( quaternion<float,AutoAlign>() ));
    CALL_SUBTEST_1( check_const_correctness(Quaternionf()) );
    CALL_SUBTEST_2(( quaternion<double,AutoAlign>() ));
    CALL_SUBTEST_2( check_const_correctness(Quaterniond()) );
    CALL_SUBTEST_3(( quaternion<float,DontAlign>() ));
    CALL_SUBTEST_4(( quaternion<double,DontAlign>() ));
    CALL_SUBTEST_5(( quaternionAlignment<float>() ));
    CALL_SUBTEST_6(( quaternionAlignment<double>() ));
    CALL_SUBTEST_1( mapQuaternion<float>() );
    CALL_SUBTEST_2( mapQuaternion<double>() );
  }
}
