// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

void test_array_of_string()
{
  typedef Array<std::string,1,Dynamic> ArrayXs;
  ArrayXs a1(3), a2(3), a3(3), a3ref(3);
  a1 << "one", "two", "three";
  a2 << "1", "2", "3";
  a3ref << "one (1)", "two (2)", "three (3)";
  std::stringstream s1;
  s1 << a1;
  VERIFY_IS_EQUAL(s1.str(), std::string("  one    two  three"));
  a3 = a1 + std::string(" (") + a2 + std::string(")");
  VERIFY((a3==a3ref).all());

  a3 = a1;
  a3 += std::string(" (") + a2 + std::string(")");
  VERIFY((a3==a3ref).all());

  a1.swap(a3);
  VERIFY((a1==a3ref).all());
  VERIFY((a3!=a3ref).all());
}
