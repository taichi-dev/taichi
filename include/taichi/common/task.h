/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <vector>
#include <string>

TC_NAMESPACE_BEGIN

class Task : public Unit {
 public:
  virtual void run(const std::vector<std::string> &parameters) {
    assert_info(parameters.size() == 0, "No parameters supported.");
    this->run();
  }
  virtual void run() {
    this->run(std::vector<std::string>());
  }
};

TC_INTERFACE(Task)

TC_NAMESPACE_END
