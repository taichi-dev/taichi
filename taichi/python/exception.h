/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/interface.h"

#include <exception>

TI_NAMESPACE_BEGIN

class ExceptionForPython : public std::exception {
 private:
  std::string msg;

 public:
  ExceptionForPython(const std::string &msg) : msg(msg) {
  }
  char const *what() const throw() {
    return msg.c_str();
  }
};

void raise_assertion_failure_in_python(const std::string &msg);

TI_NAMESPACE_END
