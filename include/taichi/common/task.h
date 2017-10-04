/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <vector>
#include <string>

TC_NAMESPACE_BEGIN

class Task : public Unit {
 public:
  virtual void run(const std::vector<std::string> &parameters) { this->run(); }
  virtual void run() {}
};

TC_INTERFACE(Task)

TC_NAMESPACE_END
