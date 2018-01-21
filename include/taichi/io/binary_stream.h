/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <cstdio>

TC_NAMESPACE_BEGIN

class BinaryFileStreamInput final {
 private:
  FILE *f;

 public:
  BinaryFileStreamInput(const std::string &fn) {
    f = std::fopen(fn.c_str(), "rb");
  }

  template <typename T>
  BinaryFileStreamInput operator>>(const T &t) {
  }

  BinaryFileStreamInput() {
    std::fclose(f);
  }
};

class BinaryFileStreamOutput final {
 private:
  FILE *f;

 public:
  BinaryFileStreamOutput(const std::string &fn) {
    f = std::fopen(fn.c_str(), "wb");
  }

  BinaryFileStreamOutput() {
    std::fclose(f);
  }
};

TC_NAMESPACE_END