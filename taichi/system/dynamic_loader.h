/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

class DynamicLoader {
 private:
  void load_dll(const std::string &dll_path);

  void close_dll();

 public:
  explicit DynamicLoader(const std::string &dll_path);

  void *load_function(const std::string &func_name);

  template <typename T>
  void load_function(const std::string &func_name, T &f) {
    f = (T)load_function(func_name);
  }

  bool loaded() const;

  ~DynamicLoader();

 private:
  void *dll = nullptr;
};

TI_NAMESPACE_END
