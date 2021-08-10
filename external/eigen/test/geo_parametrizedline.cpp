// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>

template<typename LineType> void parametrizedline(const LineType& _line)
{
  /* this test covers the following files:
     ParametrizedLine.h
  */
  using std::abs;
  const Index dim = _line.dim();
  typedef typename LineType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, LineType::AmbientDimAtCompileTime, 1> VectorType;
  typedef Hyperplane<Scalar,LineType::AmbientDimAtCompileTime> HyperplaneType;

  VectorType p0 = VectorType::Random(dim);
  VectorType p1 = VectorType::Random(dim);

  VectorType d0 = VectorType::Random(dim).normalized();

  LineType l0(p0, d0);

  Scalar s0 = internal::random<Scalar>();
  Scalar s1 = abs(internal::random<Scalar>());

  VERIFY_IS_MUCH_SMALLER_THAN( l0.distance(p0), RealScalar(1) );
  VERIFY_IS_MUCH_SMALLER_THAN( l0.distance(p0+s0*d0), RealScalar(1) );
  VERIFY_IS_APPROX( (l0.projection(p1)-p1).norm(), l0.distance(p1) );
  VERIFY_IS_MUCH_SMALLER_THAN( l0.distance(l0.projection(p1)), RealScalar(1) );
  VERIFY_IS_APPROX( Scalar(l0.distance((p0+s0*d0) + d0.unitOrthogonal() * s1)), s1 );

  // casting
  const int Dim = LineType::AmbientDimAtCompileTime;
  typedef typename GetDifferentType<Scalar>::type OtherScalar;
  ParametrizedLine<OtherScalar,Dim> hp1f = l0.template cast<OtherScalar>();
  VERIFY_IS_APPROX(hp1f.template cast<Scalar>(),l0);
  ParametrizedLine<Scalar,Dim> hp1d = l0.template cast<Scalar>();
  VERIFY_IS_APPROX(hp1d.template cast<Scalar>(),l0);

  // intersections
  VectorType p2 = VectorType::Random(dim);
  VectorType n2 = VectorType::Random(dim).normalized();
  HyperplaneType hp(p2,n2);
  Scalar t = l0.intersectionParameter(hp);
  VectorType pi = l0.pointAt(t);
  VERIFY_IS_MUCH_SMALLER_THAN(hp.signedDistance(pi), RealScalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(l0.distance(pi), RealScalar(1));
  VERIFY_IS_APPROX(l0.intersectionPoint(hp), pi);
}

template<typename Scalar> void parametrizedline_alignment()
{
  typedef ParametrizedLine<Scalar,4,AutoAlign> Line4a;
  typedef ParametrizedLine<Scalar,4,DontAlign> Line4u;

  EIGEN_ALIGN_MAX Scalar array1[16];
  EIGEN_ALIGN_MAX Scalar array2[16];
  EIGEN_ALIGN_MAX Scalar array3[16+1];
  Scalar* array3u = array3+1;

  Line4a *p1 = ::new(reinterpret_cast<void*>(array1)) Line4a;
  Line4u *p2 = ::new(reinterpret_cast<void*>(array2)) Line4u;
  Line4u *p3 = ::new(reinterpret_cast<void*>(array3u)) Line4u;
  
  p1->origin().setRandom();
  p1->direction().setRandom();
  *p2 = *p1;
  *p3 = *p1;

  VERIFY_IS_APPROX(p1->origin(), p2->origin());
  VERIFY_IS_APPROX(p1->origin(), p3->origin());
  VERIFY_IS_APPROX(p1->direction(), p2->direction());
  VERIFY_IS_APPROX(p1->direction(), p3->direction());
  
  #if defined(EIGEN_VECTORIZE) && EIGEN_MAX_STATIC_ALIGN_BYTES>0
  if(internal::packet_traits<Scalar>::Vectorizable && internal::packet_traits<Scalar>::size<=4)
    VERIFY_RAISES_ASSERT((::new(reinterpret_cast<void*>(array3u)) Line4a));
  #endif
}

void test_geo_parametrizedline()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( parametrizedline(ParametrizedLine<float,2>()) );
    CALL_SUBTEST_2( parametrizedline(ParametrizedLine<float,3>()) );
    CALL_SUBTEST_2( parametrizedline_alignment<float>() );
    CALL_SUBTEST_3( parametrizedline(ParametrizedLine<double,4>()) );
    CALL_SUBTEST_3( parametrizedline_alignment<double>() );
    CALL_SUBTEST_4( parametrizedline(ParametrizedLine<std::complex<double>,5>()) );
  }
}
