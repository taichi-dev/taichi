// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

typedef long long int64;

template<typename Scalar> Scalar check_in_range(Scalar x, Scalar y)
{
  Scalar r = internal::random<Scalar>(x,y);
  VERIFY(r>=x);
  if(y>=x)
  {
    VERIFY(r<=y);
  }
  return r;
}

template<typename Scalar> void check_all_in_range(Scalar x, Scalar y)
{
  Array<int,1,Dynamic> mask(y-x+1);
  mask.fill(0);
  long n = (y-x+1)*32;
  for(long k=0; k<n; ++k)
  {
    mask( check_in_range(x,y)-x )++;
  }
  for(Index i=0; i<mask.size(); ++i)
    if(mask(i)==0)
      std::cout << "WARNING: value " << x+i << " not reached." << std::endl;
  VERIFY( (mask>0).all() );
}

template<typename Scalar> void check_histogram(Scalar x, Scalar y, int bins)
{
  Array<int,1,Dynamic> hist(bins);
  hist.fill(0);
  int f = 100000;
  int n = bins*f;
  int64 range = int64(y)-int64(x);
  int divisor = int((range+1)/bins);
  assert(((range+1)%bins)==0);
  for(int k=0; k<n; ++k)
  {
    Scalar r = check_in_range(x,y);
    hist( int((int64(r)-int64(x))/divisor) )++;
  }
  VERIFY( (((hist.cast<double>()/double(f))-1.0).abs()<0.02).all() );
}

void test_rand()
{
  long long_ref = NumTraits<long>::highest()/10;
  signed char char_offset = (std::min)(g_repeat,64);
  signed char short_offset = (std::min)(g_repeat,16000);

  for(int i = 0; i < g_repeat*10000; i++) {
    CALL_SUBTEST(check_in_range<float>(10,11));
    CALL_SUBTEST(check_in_range<float>(1.24234523,1.24234523));
    CALL_SUBTEST(check_in_range<float>(-1,1));
    CALL_SUBTEST(check_in_range<float>(-1432.2352,-1432.2352));

    CALL_SUBTEST(check_in_range<double>(10,11));
    CALL_SUBTEST(check_in_range<double>(1.24234523,1.24234523));
    CALL_SUBTEST(check_in_range<double>(-1,1));
    CALL_SUBTEST(check_in_range<double>(-1432.2352,-1432.2352));

    CALL_SUBTEST(check_in_range<int>(0,-1));
    CALL_SUBTEST(check_in_range<short>(0,-1));
    CALL_SUBTEST(check_in_range<long>(0,-1));
    CALL_SUBTEST(check_in_range<int>(-673456,673456));
    CALL_SUBTEST(check_in_range<int>(-RAND_MAX+10,RAND_MAX-10));
    CALL_SUBTEST(check_in_range<short>(-24345,24345));
    CALL_SUBTEST(check_in_range<long>(-long_ref,long_ref));
  }

  CALL_SUBTEST(check_all_in_range<signed char>(11,11));
  CALL_SUBTEST(check_all_in_range<signed char>(11,11+char_offset));
  CALL_SUBTEST(check_all_in_range<signed char>(-5,5));
  CALL_SUBTEST(check_all_in_range<signed char>(-11-char_offset,-11));
  CALL_SUBTEST(check_all_in_range<signed char>(-126,-126+char_offset));
  CALL_SUBTEST(check_all_in_range<signed char>(126-char_offset,126));
  CALL_SUBTEST(check_all_in_range<signed char>(-126,126));

  CALL_SUBTEST(check_all_in_range<short>(11,11));
  CALL_SUBTEST(check_all_in_range<short>(11,11+short_offset));
  CALL_SUBTEST(check_all_in_range<short>(-5,5));
  CALL_SUBTEST(check_all_in_range<short>(-11-short_offset,-11));
  CALL_SUBTEST(check_all_in_range<short>(-24345,-24345+short_offset));
  CALL_SUBTEST(check_all_in_range<short>(24345,24345+short_offset));

  CALL_SUBTEST(check_all_in_range<int>(11,11));
  CALL_SUBTEST(check_all_in_range<int>(11,11+g_repeat));
  CALL_SUBTEST(check_all_in_range<int>(-5,5));
  CALL_SUBTEST(check_all_in_range<int>(-11-g_repeat,-11));
  CALL_SUBTEST(check_all_in_range<int>(-673456,-673456+g_repeat));
  CALL_SUBTEST(check_all_in_range<int>(673456,673456+g_repeat));

  CALL_SUBTEST(check_all_in_range<long>(11,11));
  CALL_SUBTEST(check_all_in_range<long>(11,11+g_repeat));
  CALL_SUBTEST(check_all_in_range<long>(-5,5));
  CALL_SUBTEST(check_all_in_range<long>(-11-g_repeat,-11));
  CALL_SUBTEST(check_all_in_range<long>(-long_ref,-long_ref+g_repeat));
  CALL_SUBTEST(check_all_in_range<long>( long_ref, long_ref+g_repeat));

  CALL_SUBTEST(check_histogram<int>(-5,5,11));
  int bins = 100;
  CALL_SUBTEST(check_histogram<int>(-3333,-3333+bins*(3333/bins)-1,bins));
  bins = 1000;
  CALL_SUBTEST(check_histogram<int>(-RAND_MAX+10,-RAND_MAX+10+bins*(RAND_MAX/bins)-1,bins));
  CALL_SUBTEST(check_histogram<int>(-RAND_MAX+10,-int64(RAND_MAX)+10+bins*(2*int64(RAND_MAX)/bins)-1,bins));
}
