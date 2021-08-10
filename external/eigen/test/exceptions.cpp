// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// Various sanity tests with exceptions:
//  - no memory leak when a custom scalar type trow an exceptions
//  - todo: complete the list of tests!

#define EIGEN_STACK_ALLOCATION_LIMIT 100000000

#include "main.h"

struct my_exception
{
  my_exception() {}
  ~my_exception() {}
};
    
class ScalarWithExceptions
{
  public:
    ScalarWithExceptions() { init(); }
    ScalarWithExceptions(const float& _v) { init(); *v = _v; }
    ScalarWithExceptions(const ScalarWithExceptions& other) { init(); *v = *(other.v); }
    ~ScalarWithExceptions() {
      delete v;
      instances--;
    }

    void init() {
      v = new float;
      instances++;
    }

    ScalarWithExceptions operator+(const ScalarWithExceptions& other) const
    {
      countdown--;
      if(countdown<=0)
        throw my_exception();
      return ScalarWithExceptions(*v+*other.v);
    }
    
    ScalarWithExceptions operator-(const ScalarWithExceptions& other) const
    { return ScalarWithExceptions(*v-*other.v); }
    
    ScalarWithExceptions operator*(const ScalarWithExceptions& other) const
    { return ScalarWithExceptions((*v)*(*other.v)); }
    
    ScalarWithExceptions& operator+=(const ScalarWithExceptions& other)
    { *v+=*other.v; return *this; }
    ScalarWithExceptions& operator-=(const ScalarWithExceptions& other)
    { *v-=*other.v; return *this; }
    ScalarWithExceptions& operator=(const ScalarWithExceptions& other)
    { *v = *(other.v); return *this; }
  
    bool operator==(const ScalarWithExceptions& other) const
    { return *v==*other.v; }
    bool operator!=(const ScalarWithExceptions& other) const
    { return *v!=*other.v; }
    
    float* v;
    static int instances;
    static int countdown;
};

ScalarWithExceptions real(const ScalarWithExceptions &x) { return x; }
ScalarWithExceptions imag(const ScalarWithExceptions & ) { return 0; }
ScalarWithExceptions conj(const ScalarWithExceptions &x) { return x; }

int ScalarWithExceptions::instances = 0;
int ScalarWithExceptions::countdown = 0;


#define CHECK_MEMLEAK(OP) {                                 \
    ScalarWithExceptions::countdown = 100;                  \
    int before = ScalarWithExceptions::instances;           \
    bool exception_thrown = false;                         \
    try { OP; }                              \
    catch (my_exception) {                                  \
      exception_thrown = true;                              \
      VERIFY(ScalarWithExceptions::instances==before && "memory leak detected in " && EIGEN_MAKESTRING(OP)); \
    } \
    VERIFY(exception_thrown && " no exception thrown in " && EIGEN_MAKESTRING(OP)); \
  }

void memoryleak()
{
  typedef Eigen::Matrix<ScalarWithExceptions,Dynamic,1> VectorType;
  typedef Eigen::Matrix<ScalarWithExceptions,Dynamic,Dynamic> MatrixType;
  
  {
    int n = 50;
    VectorType v0(n), v1(n);
    MatrixType m0(n,n), m1(n,n), m2(n,n);
    v0.setOnes(); v1.setOnes();
    m0.setOnes(); m1.setOnes(); m2.setOnes();
    CHECK_MEMLEAK(v0 = m0 * m1 * v1);
    CHECK_MEMLEAK(m2 = m0 * m1 * m2);
    CHECK_MEMLEAK((v0+v1).dot(v0+v1));
  }
  VERIFY(ScalarWithExceptions::instances==0 && "global memory leak detected in " && EIGEN_MAKESTRING(OP)); \
}

void test_exceptions()
{
  EIGEN_TRY {
    CALL_SUBTEST( memoryleak() );
  } EIGEN_CATCH(...) {}
}
