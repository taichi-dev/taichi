/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <cxxabi.h>

TC_NAMESPACE_BEGIN

// From https://en.wikipedia.org/wiki/Name_mangling

class Demangling : public Task {
  virtual void run(const std::vector<std::string> &parameters) {
    if (parameters.size() == 0) {
      printf("There should be at least one parameter for demangling.\n");
    }
    for (auto p : parameters) {
      char *demangled_name;
      int status = -1;
      demangled_name =
          abi::__cxa_demangle(parameters[0].c_str(), NULL, NULL, &status);
      printf("Demangled C++ Identifier: %s\n", demangled_name);
      free(demangled_name);
    }
  }
};

TC_IMPLEMENTATION(Task, Demangling, "demangle")

TC_NAMESPACE_END
