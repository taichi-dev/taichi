// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT
#include "main.h"

template<typename Scalar> void smallVectors()
{
  typedef Matrix<Scalar, 1, 2> V2;
  typedef Matrix<Scalar, 3, 1> V3;
  typedef Matrix<Scalar, 1, 4> V4;
  typedef Matrix<Scalar, Dynamic, 1> VX;
  Scalar x1 = internal::random<Scalar>(),
         x2 = internal::random<Scalar>(),
         x3 = internal::random<Scalar>(),
         x4 = internal::random<Scalar>();
  V2 v2(x1, x2);
  V3 v3(x1, x2, x3);
  V4 v4(x1, x2, x3, x4);
  VERIFY_IS_APPROX(x1, v2.x());
  VERIFY_IS_APPROX(x1, v3.x());
  VERIFY_IS_APPROX(x1, v4.x());
  VERIFY_IS_APPROX(x2, v2.y());
  VERIFY_IS_APPROX(x2, v3.y());
  VERIFY_IS_APPROX(x2, v4.y());
  VERIFY_IS_APPROX(x3, v3.z());
  VERIFY_IS_APPROX(x3, v4.z());
  VERIFY_IS_APPROX(x4, v4.w());

  if (!NumTraits<Scalar>::IsInteger)
  {
    VERIFY_RAISES_ASSERT(V3(2, 1))
    VERIFY_RAISES_ASSERT(V3(3, 2))
    VERIFY_RAISES_ASSERT(V3(Scalar(3), 1))
    VERIFY_RAISES_ASSERT(V3(3, Scalar(1)))
    VERIFY_RAISES_ASSERT(V3(Scalar(3), Scalar(1)))
    VERIFY_RAISES_ASSERT(V3(Scalar(123), Scalar(123)))

    VERIFY_RAISES_ASSERT(V4(1, 3))
    VERIFY_RAISES_ASSERT(V4(2, 4))
    VERIFY_RAISES_ASSERT(V4(1, Scalar(4)))
    VERIFY_RAISES_ASSERT(V4(Scalar(1), 4))
    VERIFY_RAISES_ASSERT(V4(Scalar(1), Scalar(4)))
    VERIFY_RAISES_ASSERT(V4(Scalar(123), Scalar(123)))

    VERIFY_RAISES_ASSERT(VX(3, 2))
    VERIFY_RAISES_ASSERT(VX(Scalar(3), 1))
    VERIFY_RAISES_ASSERT(VX(3, Scalar(1)))
    VERIFY_RAISES_ASSERT(VX(Scalar(3), Scalar(1)))
    VERIFY_RAISES_ASSERT(VX(Scalar(123), Scalar(123)))
  }
}

void test_smallvectors()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST(smallVectors<int>() );
    CALL_SUBTEST(smallVectors<float>() );
    CALL_SUBTEST(smallVectors<double>() );
  }
}
