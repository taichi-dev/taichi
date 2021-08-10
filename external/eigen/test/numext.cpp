// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename T>
void check_abs() {
  typedef typename NumTraits<T>::Real Real;
  Real zero(0);

  if(NumTraits<T>::IsSigned)
    VERIFY_IS_EQUAL(numext::abs(-T(1)), T(1));
  VERIFY_IS_EQUAL(numext::abs(T(0)), T(0));
  VERIFY_IS_EQUAL(numext::abs(T(1)), T(1));

  for(int k=0; k<g_repeat*100; ++k)
  {
    T x = internal::random<T>();
    if(!internal::is_same<T,bool>::value)
      x = x/Real(2);
    if(NumTraits<T>::IsSigned)
    {
      VERIFY_IS_EQUAL(numext::abs(x), numext::abs(-x));
      VERIFY( numext::abs(-x) >= zero );
    }
    VERIFY( numext::abs(x) >= zero );
    VERIFY_IS_APPROX( numext::abs2(x), numext::abs2(numext::abs(x)) );
  }
}

void test_numext() {
  CALL_SUBTEST( check_abs<bool>() );
  CALL_SUBTEST( check_abs<signed char>() );
  CALL_SUBTEST( check_abs<unsigned char>() );
  CALL_SUBTEST( check_abs<short>() );
  CALL_SUBTEST( check_abs<unsigned short>() );
  CALL_SUBTEST( check_abs<int>() );
  CALL_SUBTEST( check_abs<unsigned int>() );
  CALL_SUBTEST( check_abs<long>() );
  CALL_SUBTEST( check_abs<unsigned long>() );
  CALL_SUBTEST( check_abs<half>() );
  CALL_SUBTEST( check_abs<float>() );
  CALL_SUBTEST( check_abs<double>() );
  CALL_SUBTEST( check_abs<long double>() );

  CALL_SUBTEST( check_abs<std::complex<float> >() );
  CALL_SUBTEST( check_abs<std::complex<double> >() );
}
