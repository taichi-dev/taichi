/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <taichi/common/util.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

class TestMath : public Task {
 public:
  void run(const std::vector<std::string> &parameters) {
    for (auto p : parameters) {
      printf("%s\n", p.c_str());
    }
  }
};

TC_IMPLEMENTATION(Task, TestMath, "test_math");

TC_NAMESPACE_END
