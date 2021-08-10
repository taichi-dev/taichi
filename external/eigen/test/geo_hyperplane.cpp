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

template<typename HyperplaneType> void hyperplane(const HyperplaneType& _plane)
{
  /* this test covers the following files:
     Hyperplane.h
  */
  using std::abs;
  const Index dim = _plane.dim();
  enum { Options = HyperplaneType::Options };
  typedef typename HyperplaneType::Scalar Scalar;
  typedef typename HyperplaneType::RealScalar RealScalar;
  typedef Matrix<Scalar, HyperplaneType::AmbientDimAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, HyperplaneType::AmbientDimAtCompileTime,
                         HyperplaneType::AmbientDimAtCompileTime> MatrixType;

  VectorType p0 = VectorType::Random(dim);
  VectorType p1 = VectorType::Random(dim);

  VectorType n0 = VectorType::Random(dim).normalized();
  VectorType n1 = VectorType::Random(dim).normalized();

  HyperplaneType pl0(n0, p0);
  HyperplaneType pl1(n1, p1);
  HyperplaneType pl2 = pl1;

  Scalar s0 = internal::random<Scalar>();
  Scalar s1 = internal::random<Scalar>();

  VERIFY_IS_APPROX( n1.dot(n1), Scalar(1) );

  VERIFY_IS_MUCH_SMALLER_THAN( pl0.absDistance(p0), Scalar(1) );
  if(numext::abs2(s0)>RealScalar(1e-6))
    VERIFY_IS_APPROX( pl1.signedDistance(p1 + n1 * s0), s0);
  else
    VERIFY_IS_MUCH_SMALLER_THAN( abs(pl1.signedDistance(p1 + n1 * s0) - s0), Scalar(1) );
  VERIFY_IS_MUCH_SMALLER_THAN( pl1.signedDistance(pl1.projection(p0)), Scalar(1) );
  VERIFY_IS_MUCH_SMALLER_THAN( pl1.absDistance(p1 +  pl1.normal().unitOrthogonal() * s1), Scalar(1) );

  // transform
  if (!NumTraits<Scalar>::IsComplex)
  {
    MatrixType rot = MatrixType::Random(dim,dim).householderQr().householderQ();
    DiagonalMatrix<Scalar,HyperplaneType::AmbientDimAtCompileTime> scaling(VectorType::Random());
    Translation<Scalar,HyperplaneType::AmbientDimAtCompileTime> translation(VectorType::Random());
    
    while(scaling.diagonal().cwiseAbs().minCoeff()<RealScalar(1e-4)) scaling.diagonal() = VectorType::Random();

    pl2 = pl1;
    VERIFY_IS_MUCH_SMALLER_THAN( pl2.transform(rot).absDistance(rot * p1), Scalar(1) );
    pl2 = pl1;
    VERIFY_IS_MUCH_SMALLER_THAN( pl2.transform(rot,Isometry).absDistance(rot * p1), Scalar(1) );
    pl2 = pl1;
    VERIFY_IS_MUCH_SMALLER_THAN( pl2.transform(rot*scaling).absDistance((rot*scaling) * p1), Scalar(1) );
    VERIFY_IS_APPROX( pl2.normal().norm(), RealScalar(1) );
    pl2 = pl1;
    VERIFY_IS_MUCH_SMALLER_THAN( pl2.transform(rot*scaling*translation)
                                  .absDistance((rot*scaling*translation) * p1), Scalar(1) );
    VERIFY_IS_APPROX( pl2.normal().norm(), RealScalar(1) );
    pl2 = pl1;
    VERIFY_IS_MUCH_SMALLER_THAN( pl2.transform(rot*translation,Isometry)
                                 .absDistance((rot*translation) * p1), Scalar(1) );
    VERIFY_IS_APPROX( pl2.normal().norm(), RealScalar(1) );
  }

  // casting
  const int Dim = HyperplaneType::AmbientDimAtCompileTime;
  typedef typename GetDifferentType<Scalar>::type OtherScalar;
  Hyperplane<OtherScalar,Dim,Options> hp1f = pl1.template cast<OtherScalar>();
  VERIFY_IS_APPROX(hp1f.template cast<Scalar>(),pl1);
  Hyperplane<Scalar,Dim,Options> hp1d = pl1.template cast<Scalar>();
  VERIFY_IS_APPROX(hp1d.template cast<Scalar>(),pl1);
}

template<typename Scalar> void lines()
{
  using std::abs;
  typedef Hyperplane<Scalar, 2> HLine;
  typedef ParametrizedLine<Scalar, 2> PLine;
  typedef Matrix<Scalar,2,1> Vector;
  typedef Matrix<Scalar,3,1> CoeffsType;

  for(int i = 0; i < 10; i++)
  {
    Vector center = Vector::Random();
    Vector u = Vector::Random();
    Vector v = Vector::Random();
    Scalar a = internal::random<Scalar>();
    while (abs(a-1) < Scalar(1e-4)) a = internal::random<Scalar>();
    while (u.norm() < Scalar(1e-4)) u = Vector::Random();
    while (v.norm() < Scalar(1e-4)) v = Vector::Random();

    HLine line_u = HLine::Through(center + u, center + a*u);
    HLine line_v = HLine::Through(center + v, center + a*v);

    // the line equations should be normalized so that a^2+b^2=1
    VERIFY_IS_APPROX(line_u.normal().norm(), Scalar(1));
    VERIFY_IS_APPROX(line_v.normal().norm(), Scalar(1));

    Vector result = line_u.intersection(line_v);

    // the lines should intersect at the point we called "center"
    if(abs(a-1) > Scalar(1e-2) && abs(v.normalized().dot(u.normalized()))<Scalar(0.9))
      VERIFY_IS_APPROX(result, center);

    // check conversions between two types of lines
    PLine pl(line_u); // gcc 3.3 will commit suicide if we don't name this variable
    HLine line_u2(pl);
    CoeffsType converted_coeffs = line_u2.coeffs();
    if(line_u2.normal().dot(line_u.normal())<Scalar(0))
      converted_coeffs = -line_u2.coeffs();
    VERIFY(line_u.coeffs().isApprox(converted_coeffs));
  }
}

template<typename Scalar> void planes()
{
  using std::abs;
  typedef Hyperplane<Scalar, 3> Plane;
  typedef Matrix<Scalar,3,1> Vector;

  for(int i = 0; i < 10; i++)
  {
    Vector v0 = Vector::Random();
    Vector v1(v0), v2(v0);
    if(internal::random<double>(0,1)>0.25)
      v1 += Vector::Random();
    if(internal::random<double>(0,1)>0.25)
      v2 += v1 * std::pow(internal::random<Scalar>(0,1),internal::random<int>(1,16));
    if(internal::random<double>(0,1)>0.25)
      v2 += Vector::Random() * std::pow(internal::random<Scalar>(0,1),internal::random<int>(1,16));

    Plane p0 = Plane::Through(v0, v1, v2);

    VERIFY_IS_APPROX(p0.normal().norm(), Scalar(1));
    VERIFY_IS_MUCH_SMALLER_THAN(p0.absDistance(v0), Scalar(1));
    VERIFY_IS_MUCH_SMALLER_THAN(p0.absDistance(v1), Scalar(1));
    VERIFY_IS_MUCH_SMALLER_THAN(p0.absDistance(v2), Scalar(1));
  }
}

template<typename Scalar> void hyperplane_alignment()
{
  typedef Hyperplane<Scalar,3,AutoAlign> Plane3a;
  typedef Hyperplane<Scalar,3,DontAlign> Plane3u;

  EIGEN_ALIGN_MAX Scalar array1[4];
  EIGEN_ALIGN_MAX Scalar array2[4];
  EIGEN_ALIGN_MAX Scalar array3[4+1];
  Scalar* array3u = array3+1;

  Plane3a *p1 = ::new(reinterpret_cast<void*>(array1)) Plane3a;
  Plane3u *p2 = ::new(reinterpret_cast<void*>(array2)) Plane3u;
  Plane3u *p3 = ::new(reinterpret_cast<void*>(array3u)) Plane3u;
  
  p1->coeffs().setRandom();
  *p2 = *p1;
  *p3 = *p1;

  VERIFY_IS_APPROX(p1->coeffs(), p2->coeffs());
  VERIFY_IS_APPROX(p1->coeffs(), p3->coeffs());
  
  #if defined(EIGEN_VECTORIZE) && EIGEN_MAX_STATIC_ALIGN_BYTES > 0
  if(internal::packet_traits<Scalar>::Vectorizable && internal::packet_traits<Scalar>::size<=4)
    VERIFY_RAISES_ASSERT((::new(reinterpret_cast<void*>(array3u)) Plane3a));
  #endif
}


void test_geo_hyperplane()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( hyperplane(Hyperplane<float,2>()) );
    CALL_SUBTEST_2( hyperplane(Hyperplane<float,3>()) );
    CALL_SUBTEST_2( hyperplane(Hyperplane<float,3,DontAlign>()) );
    CALL_SUBTEST_2( hyperplane_alignment<float>() );
    CALL_SUBTEST_3( hyperplane(Hyperplane<double,4>()) );
    CALL_SUBTEST_4( hyperplane(Hyperplane<std::complex<double>,5>()) );
    CALL_SUBTEST_1( lines<float>() );
    CALL_SUBTEST_3( lines<double>() );
    CALL_SUBTEST_2( planes<float>() );
    CALL_SUBTEST_5( planes<double>() );
  }
}
