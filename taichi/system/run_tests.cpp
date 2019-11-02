/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

class RunTests : public Task {
  virtual std::string run(const std::vector<std::string> &parameters) {
    run_tests(parameters);
    return "";
  }
};

TC_IMPLEMENTATION(Task, RunTests, "test");

TC_NAMESPACE_END
