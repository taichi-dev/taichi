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

std::string cpp_demangle(const std::string &mangled_name) {
  char *demangled_name;
  int status = -1;
  demangled_name =
      abi::__cxa_demangle(mangled_name.c_str(), NULL, NULL, &status);
  std::string ret(demangled_name);
  free(demangled_name);
  return ret;
}

class Demangling : public Task {
  virtual std::string run(const std::vector<std::string> &parameters) {
    if (parameters.size() == 0) {
      printf("There should be at least one parameter for demangling.\n");
    }
    for (auto p : parameters) {
#if !defined(_WIN64)
      printf("Demangled C++ Identifier: %s\n", cpp_demangle(p).c_str());
#else
      TC_NOT_IMPLEMENTED
#endif
    }
    return "";
  }
};

TC_IMPLEMENTATION(Task, Demangling, "demangle")

TC_NAMESPACE_END
