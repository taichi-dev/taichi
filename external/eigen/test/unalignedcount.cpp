// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

static int nb_load;
static int nb_loadu;
static int nb_store;
static int nb_storeu;

#define EIGEN_DEBUG_ALIGNED_LOAD    { nb_load++;    }
#define EIGEN_DEBUG_UNALIGNED_LOAD  { nb_loadu++;   }
#define EIGEN_DEBUG_ALIGNED_STORE   { nb_store++;   }
#define EIGEN_DEBUG_UNALIGNED_STORE { nb_storeu++;  }

#define VERIFY_ALIGNED_UNALIGNED_COUNT(XPR,AL,UL,AS,US) {\
    nb_load = nb_loadu = nb_store = nb_storeu = 0; \
    XPR; \
    if(!(nb_load==AL && nb_loadu==UL && nb_store==AS && nb_storeu==US)) \
      std::cerr << " >> " << nb_load << ", " << nb_loadu << ", " << nb_store << ", " << nb_storeu << "\n"; \
    VERIFY( (#XPR) && nb_load==AL && nb_loadu==UL && nb_store==AS && nb_storeu==US ); \
  }


#include "main.h"

void test_unalignedcount()
{
  #if defined(EIGEN_VECTORIZE_AVX)
  VectorXf a(40), b(40);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a += b, 10, 0, 5, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) += b.segment(0,40), 5, 5, 5, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) -= b.segment(0,40), 5, 5, 5, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) *= 3.5, 5, 0, 5, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) /= 3.5, 5, 0, 5, 0);
  #elif defined(EIGEN_VECTORIZE_SSE)
  VectorXf a(40), b(40);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a += b, 20, 0, 10, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) += b.segment(0,40), 10, 10, 10, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) -= b.segment(0,40), 10, 10, 10, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) *= 3.5, 10, 0, 10, 0);
  VERIFY_ALIGNED_UNALIGNED_COUNT(a.segment(0,40) /= 3.5, 10, 0, 10, 0);
  #else
  // The following line is to eliminate "variable not used" warnings
  nb_load = nb_loadu = nb_store = nb_storeu = 0;
  int a(0), b(0);
  VERIFY(a==b);
  #endif
}
