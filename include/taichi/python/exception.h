/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <exception>

TC_NAMESPACE_BEGIN

class ExceptionForPython : std::exception {
 private:
  std::string msg;

 public:
  ExceptionForPython(const std::string &msg) : msg(msg) {}
  char const *what() const throw() { return msg.c_str(); }
};

void raise_assertion_failure_in_python(const std::string &msg);

TC_NAMESPACE_END
