// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "utilities.h"
#include "tensor_interface.hh"
#include "bench.hh"
#include "basic_actions.hh"

BTL_MAIN;

int main()
{
  bench<Action_axpy<tensor_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);
  bench<Action_axpby<tensor_interface<REAL_TYPE> > >(MIN_AXPY,MAX_AXPY,NB_POINT);

  return 0;
}
