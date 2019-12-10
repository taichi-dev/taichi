/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/system/threading.h>

TC_NAMESPACE_BEGIN

class ThreadPool {
public:
  ThreadPool() {
    TC_TAG;
  }
};

bool test_threading() {
  auto tp = ThreadPool();
  return true;
}

TC_NAMESPACE_END
