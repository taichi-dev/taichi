// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "../Eigen/SpecialFunctions"

template<typename X, typename Y>
void verify_component_wise(const X& x, const Y& y)
{
  for(Index i=0; i<x.size(); ++i)
  {
    if((numext::isfinite)(y(i)))
      VERIFY_IS_APPROX( x(i), y(i) );
    else if((numext::isnan)(y(i)))
      VERIFY((numext::isnan)(x(i)));
    else
      VERIFY_IS_EQUAL( x(i), y(i) );
  }
}

template<typename ArrayType> void array_special_functions()
{
  using std::abs;
  using std::sqrt;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Scalar plusinf = std::numeric_limits<Scalar>::infinity();
  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

  Index rows = internal::random<Index>(1,30);
  Index cols = 1;

  // API
  {
    ArrayType m1 = ArrayType::Random(rows,cols);
#if EIGEN_HAS_C99_MATH
    VERIFY_IS_APPROX(m1.lgamma(), lgamma(m1));
    VERIFY_IS_APPROX(m1.digamma(), digamma(m1));
    VERIFY_IS_APPROX(m1.erf(), erf(m1));
    VERIFY_IS_APPROX(m1.erfc(), erfc(m1));
#endif  // EIGEN_HAS_C99_MATH
  }


#if EIGEN_HAS_C99_MATH
  // check special functions (comparing against numpy implementation)
  if (!NumTraits<Scalar>::IsComplex)
  {

    {
      ArrayType m1 = ArrayType::Random(rows,cols);
      ArrayType m2 = ArrayType::Random(rows,cols);

      // Test various propreties of igamma & igammac.  These are normalized
      // gamma integrals where
      //   igammac(a, x) = Gamma(a, x) / Gamma(a)
      //   igamma(a, x) = gamma(a, x) / Gamma(a)
      // where Gamma and gamma are considered the standard unnormalized
      // upper and lower incomplete gamma functions, respectively.
      ArrayType a = m1.abs() + 2;
      ArrayType x = m2.abs() + 2;
      ArrayType zero = ArrayType::Zero(rows, cols);
      ArrayType one = ArrayType::Constant(rows, cols, Scalar(1.0));
      ArrayType a_m1 = a - one;
      ArrayType Gamma_a_x = Eigen::igammac(a, x) * a.lgamma().exp();
      ArrayType Gamma_a_m1_x = Eigen::igammac(a_m1, x) * a_m1.lgamma().exp();
      ArrayType gamma_a_x = Eigen::igamma(a, x) * a.lgamma().exp();
      ArrayType gamma_a_m1_x = Eigen::igamma(a_m1, x) * a_m1.lgamma().exp();

      // Gamma(a, 0) == Gamma(a)
      VERIFY_IS_APPROX(Eigen::igammac(a, zero), one);

      // Gamma(a, x) + gamma(a, x) == Gamma(a)
      VERIFY_IS_APPROX(Gamma_a_x + gamma_a_x, a.lgamma().exp());

      // Gamma(a, x) == (a - 1) * Gamma(a-1, x) + x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(Gamma_a_x, (a - 1) * Gamma_a_m1_x + x.pow(a-1) * (-x).exp());

      // gamma(a, x) == (a - 1) * gamma(a-1, x) - x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(gamma_a_x, (a - 1) * gamma_a_m1_x - x.pow(a-1) * (-x).exp());
    }

    {
      // Check exact values of igamma and igammac against a third party calculation.
      Scalar a_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};
      Scalar x_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};

      // location i*6+j corresponds to a_s[i], x_s[j].
      Scalar igamma_s[][6] = {{0.0, nan, nan, nan, nan, nan},
                              {0.0, 0.6321205588285578, 0.7768698398515702,
                              0.9816843611112658, 9.999500016666262e-05, 1.0},
                              {0.0, 0.4275932955291202, 0.608374823728911,
                              0.9539882943107686, 7.522076445089201e-07, 1.0},
                              {0.0, 0.01898815687615381, 0.06564245437845008,
                              0.5665298796332909, 4.166333347221828e-18, 1.0},
                              {0.0, 0.9999780593618628, 0.9999899967080838,
                              0.9999996219837988, 0.9991370418689945, 1.0},
                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5042041932513908}};
      Scalar igammac_s[][6] = {{nan, nan, nan, nan, nan, nan},
                              {1.0, 0.36787944117144233, 0.22313016014842982,
                                0.018315638888734182, 0.9999000049998333, 0.0},
                              {1.0, 0.5724067044708798, 0.3916251762710878,
                                0.04601170568923136, 0.9999992477923555, 0.0},
                              {1.0, 0.9810118431238462, 0.9343575456215499,
                                0.4334701203667089, 1.0, 0.0},
                              {1.0, 2.1940638138146658e-05, 1.0003291916285e-05,
                                3.7801620118431334e-07, 0.0008629581310054535,
                                0.0},
                              {1.0, 1.0, 1.0, 1.0, 1.0, 0.49579580674813944}};
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          if ((std::isnan)(igamma_s[i][j])) {
            VERIFY((std::isnan)(numext::igamma(a_s[i], x_s[j])));
          } else {
            VERIFY_IS_APPROX(numext::igamma(a_s[i], x_s[j]), igamma_s[i][j]);
          }

          if ((std::isnan)(igammac_s[i][j])) {
            VERIFY((std::isnan)(numext::igammac(a_s[i], x_s[j])));
          } else {
            VERIFY_IS_APPROX(numext::igammac(a_s[i], x_s[j]), igammac_s[i][j]);
          }
        }
      }
    }
  }
#endif  // EIGEN_HAS_C99_MATH

  // Check the zeta function against scipy.special.zeta
  {
    ArrayType x(7), q(7), res(7), ref(7);
    x << 1.5,   4, 10.5, 10000.5,    3, 1,        0.9;
    q << 2,   1.5,    3,  1.0001, -2.5, 1.2345, 1.2345;
    ref << 1.61237534869, 0.234848505667, 1.03086757337e-5, 0.367879440865, 0.054102025820864097, plusinf, nan;
    CALL_SUBTEST( verify_component_wise(ref, ref); );
    CALL_SUBTEST( res = x.zeta(q); verify_component_wise(res, ref); );
    CALL_SUBTEST( res = zeta(x,q); verify_component_wise(res, ref); );
  }

  // digamma
  {
    ArrayType x(7), res(7), ref(7);
    x << 1, 1.5, 4, -10.5, 10000.5, 0, -1;
    ref << -0.5772156649015329, 0.03648997397857645, 1.2561176684318, 2.398239129535781, 9.210340372392849, plusinf, plusinf;
    CALL_SUBTEST( verify_component_wise(ref, ref); );

    CALL_SUBTEST( res = x.digamma(); verify_component_wise(res, ref); );
    CALL_SUBTEST( res = digamma(x);  verify_component_wise(res, ref); );
  }


#if EIGEN_HAS_C99_MATH
  {
    ArrayType n(11), x(11), res(11), ref(11);
    n << 1, 1,    1, 1.5,   17,   31,   28,    8, 42, 147, 170;
    x << 2, 3, 25.5, 1.5,  4.7, 11.8, 17.7, 30.2, 15.8, 54.1, 64;
    ref << 0.644934066848, 0.394934066848, 0.0399946696496, nan, 293.334565435, 0.445487887616, -2.47810300902e-07, -8.29668781082e-09, -0.434562276666, 0.567742190178, -0.0108615497927;
    CALL_SUBTEST( verify_component_wise(ref, ref); );

    if(sizeof(RealScalar)>=8) {  // double
      // Reason for commented line: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1232
      //       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res, ref); );
      CALL_SUBTEST( res = polygamma(n,x);  verify_component_wise(res, ref); );
    }
    else {
      //       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res.head(8), ref.head(8)); );
      CALL_SUBTEST( res = polygamma(n,x); verify_component_wise(res.head(8), ref.head(8)); );
    }
  }
#endif

#if EIGEN_HAS_C99_MATH
  {
    // Inputs and ground truth generated with scipy via:
    //   a = np.logspace(-3, 3, 5) - 1e-3
    //   b = np.logspace(-3, 3, 5) - 1e-3
    //   x = np.linspace(-0.1, 1.1, 5)
    //   (full_a, full_b, full_x) = np.vectorize(lambda a, b, x: (a, b, x))(*np.ix_(a, b, x))
    //   full_a = full_a.flatten().tolist()  # same for full_b, full_x
    //   v = scipy.special.betainc(full_a, full_b, full_x).flatten().tolist()
    //
    // Note in Eigen, we call betainc with arguments in the order (x, a, b).
    ArrayType a(125);
    ArrayType b(125);
    ArrayType x(125);
    ArrayType v(125);
    ArrayType res(125);

    a << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
        0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
        0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
        999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
        999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999, 999.999,
        999.999, 999.999, 999.999;

    b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379, 0.999,
        0.999, 0.999, 0.999, 0.999, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 31.62177660168379, 999.999,
        999.999, 999.999, 999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.999, 0.999, 0.999, 0.999,
        0.999, 31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999,
        999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999,
        999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999,
        999.999, 999.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03062277660168379,
        0.03062277660168379, 0.03062277660168379, 0.03062277660168379,
        0.03062277660168379, 0.999, 0.999, 0.999, 0.999, 0.999,
        31.62177660168379, 31.62177660168379, 31.62177660168379,
        31.62177660168379, 31.62177660168379, 999.999, 999.999, 999.999,
        999.999, 999.999;

    x << -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
        0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2,
        0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1,
        0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1,
        -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8,
        1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
        0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2,
        0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1,
        0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5, 0.8, 1.1, -0.1, 0.2, 0.5,
        0.8, 1.1;

    v << nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
        nan, nan, nan, 0.47972119876364683, 0.5, 0.5202788012363533, nan, nan,
        0.9518683957740043, 0.9789663010413743, 0.9931729188073435, nan, nan,
        0.999995949033062, 0.9999999999993698, 0.9999999999999999, nan, nan,
        0.9999999999999999, 0.9999999999999999, 0.9999999999999999, nan, nan,
        nan, nan, nan, nan, nan, 0.006827081192655869, 0.0210336989586256,
        0.04813160422599567, nan, nan, 0.20014344256217678, 0.5000000000000001,
        0.7998565574378232, nan, nan, 0.9991401428435834, 0.999999999698403,
        0.9999999999999999, nan, nan, 0.9999999999999999, 0.9999999999999999,
        0.9999999999999999, nan, nan, nan, nan, nan, nan, nan,
        1.0646600232370887e-25, 6.301722877826246e-13, 4.050966937974938e-06,
        nan, nan, 7.864342668429763e-23, 3.015969667594166e-10,
        0.0008598571564165444, nan, nan, 6.031987710123844e-08,
        0.5000000000000007, 0.9999999396801229, nan, nan, 0.9999999999999999,
        0.9999999999999999, 0.9999999999999999, nan, nan, nan, nan, nan, nan,
        nan, 0.0, 7.029920380986636e-306, 2.2450728208591345e-101, nan, nan,
        0.0, 9.275871147869727e-302, 1.2232913026152827e-97, nan, nan, 0.0,
        3.0891393081932924e-252, 2.9303043666183996e-60, nan, nan,
        2.248913486879199e-196, 0.5000000000004947, 0.9999999999999999, nan;

    CALL_SUBTEST(res = betainc(a, b, x);
                 verify_component_wise(res, v););
  }

  // Test various properties of betainc
  {
    ArrayType m1 = ArrayType::Random(32);
    ArrayType m2 = ArrayType::Random(32);
    ArrayType m3 = ArrayType::Random(32);
    ArrayType one = ArrayType::Constant(32, Scalar(1.0));
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    ArrayType a = (m1 * 4.0).exp();
    ArrayType b = (m2 * 4.0).exp();
    ArrayType x = m3.abs();

    // betainc(a, 1, x) == x**a
    CALL_SUBTEST(
        ArrayType test = betainc(a, one, x);
        ArrayType expected = x.pow(a);
        verify_component_wise(test, expected););

    // betainc(1, b, x) == 1 - (1 - x)**b
    CALL_SUBTEST(
        ArrayType test = betainc(one, b, x);
        ArrayType expected = one - (one - x).pow(b);
        verify_component_wise(test, expected););

    // betainc(a, b, x) == 1 - betainc(b, a, 1-x)
    CALL_SUBTEST(
        ArrayType test = betainc(a, b, x) + betainc(b, a, one - x);
        ArrayType expected = one;
        verify_component_wise(test, expected););

    // betainc(a+1, b, x) = betainc(a, b, x) - x**a * (1 - x)**b / (a * beta(a, b))
    CALL_SUBTEST(
        ArrayType num = x.pow(a) * (one - x).pow(b);
        ArrayType denom = a * (a.lgamma() + b.lgamma() - (a + b).lgamma()).exp();
        // Add eps to rhs and lhs so that component-wise test doesn't result in
        // nans when both outputs are zeros.
        ArrayType expected = betainc(a, b, x) - num / denom + eps;
        ArrayType test = betainc(a + one, b, x) + eps;
        if (sizeof(Scalar) >= 8) { // double
          verify_component_wise(test, expected);
        } else {
          // Reason for limited test: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1232
          verify_component_wise(test.head(8), expected.head(8));
        });

    // betainc(a, b+1, x) = betainc(a, b, x) + x**a * (1 - x)**b / (b * beta(a, b))
    CALL_SUBTEST(
        // Add eps to rhs and lhs so that component-wise test doesn't result in
        // nans when both outputs are zeros.
        ArrayType num = x.pow(a) * (one - x).pow(b);
        ArrayType denom = b * (a.lgamma() + b.lgamma() - (a + b).lgamma()).exp();
        ArrayType expected = betainc(a, b, x) + num / denom + eps;
        ArrayType test = betainc(a, b + one, x) + eps;
        verify_component_wise(test, expected););
  }
#endif
}

void test_special_functions()
{
  CALL_SUBTEST_1(array_special_functions<ArrayXf>());
  CALL_SUBTEST_2(array_special_functions<ArrayXd>());
}
