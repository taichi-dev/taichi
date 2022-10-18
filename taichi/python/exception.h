/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"

#include <exception>

namespace taichi {

class ExceptionForPython : public std::exception {
 private:
  std::string msg_;

 public:
  explicit ExceptionForPython(const std::string &msg) : msg_(msg) {
  }
  char const *what() const noexcept override {
    return msg_.c_str();
  }
};

void raise_assertion_failure_in_python(const std::string &msg);

}  // namespace taichi
