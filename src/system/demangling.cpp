/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#if !defined(_WIN64)
#include <cxxabi.h>
#endif

TC_NAMESPACE_BEGIN

// From https://en.wikipedia.org/wiki/Name_mangling

class Demangling : public Task {
  virtual void run(const std::vector<std::string> &parameters) {
    if (parameters.size() == 0) {
      printf("There should be at least one parameter for demangling.\n");
    }
    for (auto p : parameters) {
#if !defined(_WIN64)
      char *demangled_name;
      int status = -1;
      demangled_name =
          abi::__cxa_demangle(parameters[0].c_str(), NULL, NULL, &status);
      printf("Demangled C++ Identifier: %s\n", demangled_name);
      free(demangled_name);
#else
      TC_NOT_IMPLEMENTED
#endif
    }
  }
};

TC_IMPLEMENTATION(Task, Demangling, "demangle")

TC_NAMESPACE_END
