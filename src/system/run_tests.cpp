/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <cxxabi.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

class RunTests : public Task {
  virtual void run(const std::vector<std::string> &parameters) {
    run_tests();
  }
};

TC_IMPLEMENTATION(Task, RunTests, "test");

TC_NAMESPACE_END
