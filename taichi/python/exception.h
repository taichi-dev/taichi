/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"

#include <exception>

TI_NAMESPACE_BEGIN

class ExceptionForPython : public std::exception {
 private:
  std::string msg_;

 public:
  ExceptionForPython(const std::string &msg) : msg_(msg) {
  }
  char const *what() const throw() override {
    return msg_.c_str();
  }
};

void raise_assertion_failure_in_python(const std::string &msg);

TI_NAMESPACE_END
