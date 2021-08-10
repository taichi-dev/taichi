// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename Scalar> void special_numbers()
{
  typedef Matrix<Scalar, Dynamic,Dynamic> MatType;
  int rows = internal::random<int>(1,300);
  int cols = internal::random<int>(1,300);
  
  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar inf = std::numeric_limits<Scalar>::infinity();
  Scalar s1 = internal::random<Scalar>();
  
  MatType m1    = MatType::Random(rows,cols),
          mnan  = MatType::Random(rows,cols),
          minf  = MatType::Random(rows,cols),
          mboth = MatType::Random(rows,cols);
          
  int n = internal::random<int>(1,10);
  for(int k=0; k<n; ++k)
  {
    mnan(internal::random<int>(0,rows-1), internal::random<int>(0,cols-1)) = nan;
    minf(internal::random<int>(0,rows-1), internal::random<int>(0,cols-1)) = inf;
  }
  mboth = mnan + minf;
  
  VERIFY(!m1.hasNaN());
  VERIFY(m1.allFinite());
  
  VERIFY(mnan.hasNaN());
  VERIFY((s1*mnan).hasNaN());
  VERIFY(!minf.hasNaN());
  VERIFY(!(2*minf).hasNaN());
  VERIFY(mboth.hasNaN());
  VERIFY(mboth.array().hasNaN());
  
  VERIFY(!mnan.allFinite());
  VERIFY(!minf.allFinite());
  VERIFY(!(minf-mboth).allFinite());
  VERIFY(!mboth.allFinite());
  VERIFY(!mboth.array().allFinite());
}

void test_special_numbers()
{
  for(int i = 0; i < 10*g_repeat; i++) {
    CALL_SUBTEST_1( special_numbers<float>() );
    CALL_SUBTEST_1( special_numbers<double>() );
  }
}
