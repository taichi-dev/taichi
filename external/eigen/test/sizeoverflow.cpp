// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#define VERIFY_THROWS_BADALLOC(a) {                           \
    bool threw = false;                                       \
    try {                                                     \
      a;                                                      \
    }                                                         \
    catch (std::bad_alloc&) { threw = true; }                 \
    VERIFY(threw && "should have thrown bad_alloc: " #a);     \
  }

template<typename MatrixType>
void triggerMatrixBadAlloc(Index rows, Index cols)
{
  VERIFY_THROWS_BADALLOC( MatrixType m(rows, cols) );
  VERIFY_THROWS_BADALLOC( MatrixType m; m.resize(rows, cols) );
  VERIFY_THROWS_BADALLOC( MatrixType m; m.conservativeResize(rows, cols) );
}

template<typename VectorType>
void triggerVectorBadAlloc(Index size)
{
  VERIFY_THROWS_BADALLOC( VectorType v(size) );
  VERIFY_THROWS_BADALLOC( VectorType v; v.resize(size) );
  VERIFY_THROWS_BADALLOC( VectorType v; v.conservativeResize(size) );
}

void test_sizeoverflow()
{
  // there are 2 levels of overflow checking. first in PlainObjectBase.h we check for overflow in rows*cols computations.
  // this is tested in tests of the form times_itself_gives_0 * times_itself_gives_0
  // Then in Memory.h we check for overflow in size * sizeof(T) computations.
  // this is tested in tests of the form times_4_gives_0 * sizeof(float)
  
  size_t times_itself_gives_0 = size_t(1) << (8 * sizeof(Index) / 2);
  VERIFY(times_itself_gives_0 * times_itself_gives_0 == 0);

  size_t times_4_gives_0 = size_t(1) << (8 * sizeof(Index) - 2);
  VERIFY(times_4_gives_0 * 4 == 0);

  size_t times_8_gives_0 = size_t(1) << (8 * sizeof(Index) - 3);
  VERIFY(times_8_gives_0 * 8 == 0);

  triggerMatrixBadAlloc<MatrixXf>(times_itself_gives_0, times_itself_gives_0);
  triggerMatrixBadAlloc<MatrixXf>(times_itself_gives_0 / 4, times_itself_gives_0);
  triggerMatrixBadAlloc<MatrixXf>(times_4_gives_0, 1);

  triggerMatrixBadAlloc<MatrixXd>(times_itself_gives_0, times_itself_gives_0);
  triggerMatrixBadAlloc<MatrixXd>(times_itself_gives_0 / 8, times_itself_gives_0);
  triggerMatrixBadAlloc<MatrixXd>(times_8_gives_0, 1);
  
  triggerVectorBadAlloc<VectorXf>(times_4_gives_0);
  
  triggerVectorBadAlloc<VectorXd>(times_8_gives_0);
}
