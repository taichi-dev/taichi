// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christoph Hertzberg <chtz@informatik.uni-bremen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/AutoDiff>

/*
 * In this file scalar derivations are tested for correctness.
 * TODO add more tests!
 */

template<typename Scalar> void check_atan2()
{
  typedef Matrix<Scalar, 1, 1> Deriv1;
  typedef AutoDiffScalar<Deriv1> AD;
  
  AD x(internal::random<Scalar>(-3.0, 3.0), Deriv1::UnitX());
  
  using std::exp;
  Scalar r = exp(internal::random<Scalar>(-10, 10));
  
  AD s = sin(x), c = cos(x);
  AD res = atan2(r*s, r*c);
  
  VERIFY_IS_APPROX(res.value(), x.value());
  VERIFY_IS_APPROX(res.derivatives(), x.derivatives());

  res = atan2(r*s+0, r*c+0);
  VERIFY_IS_APPROX(res.value(), x.value());
  VERIFY_IS_APPROX(res.derivatives(), x.derivatives());
}

template<typename Scalar> void check_hyperbolic_functions()
{
  using std::sinh;
  using std::cosh;
  using std::tanh;
  typedef Matrix<Scalar, 1, 1> Deriv1;
  typedef AutoDiffScalar<Deriv1> AD;
  Deriv1 p = Deriv1::Random();
  AD val(p.x(),Deriv1::UnitX());

  Scalar cosh_px = std::cosh(p.x());
  AD res1 = tanh(val);
  VERIFY_IS_APPROX(res1.value(), std::tanh(p.x()));
  VERIFY_IS_APPROX(res1.derivatives().x(), Scalar(1.0) / (cosh_px * cosh_px));

  AD res2 = sinh(val);
  VERIFY_IS_APPROX(res2.value(), std::sinh(p.x()));
  VERIFY_IS_APPROX(res2.derivatives().x(), cosh_px);

  AD res3 = cosh(val);
  VERIFY_IS_APPROX(res3.value(), cosh_px);
  VERIFY_IS_APPROX(res3.derivatives().x(), std::sinh(p.x()));

  // Check constant values.
  const Scalar sample_point = Scalar(1) / Scalar(3); 
  val = AD(sample_point,Deriv1::UnitX());
  res1 = tanh(val);
  VERIFY_IS_APPROX(res1.derivatives().x(), Scalar(0.896629559604914));

  res2 = sinh(val);
  VERIFY_IS_APPROX(res2.derivatives().x(), Scalar(1.056071867829939));

  res3 = cosh(val);
  VERIFY_IS_APPROX(res3.derivatives().x(), Scalar(0.339540557256150));
}

template <typename Scalar>
void check_limits_specialization()
{
  typedef Eigen::Matrix<Scalar, 1, 1> Deriv;
  typedef Eigen::AutoDiffScalar<Deriv> AD;

  typedef std::numeric_limits<AD> A;
  typedef std::numeric_limits<Scalar> B;

  // workaround "unsed typedef" warning:
  VERIFY(!bool(internal::is_same<B, A>::value));

#if EIGEN_HAS_CXX11
  VERIFY(bool(std::is_base_of<B, A>::value));
#endif
}

void test_autodiff_scalar()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( check_atan2<float>() );
    CALL_SUBTEST_2( check_atan2<double>() );
    CALL_SUBTEST_3( check_hyperbolic_functions<float>() );
    CALL_SUBTEST_4( check_hyperbolic_functions<double>() );
    CALL_SUBTEST_5( check_limits_specialization<double>());
  }
}
