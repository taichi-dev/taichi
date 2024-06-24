/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/core.h"
#include "taichi/common/task.h"
#if !defined(_WIN64)
#include <cxxabi.h>
#endif

#if defined(TI_PLATFORM_WINDOWS)
#include <DbgHelp.h>
#endif

namespace taichi {

// From https://en.wikipedia.org/wiki/Name_mangling

std::string cpp_demangle(const std::string &mangled_name) {
#if defined(TI_PLATFORM_UNIX)
  char *demangled_name;
  int status = -1;
  demangled_name =
      abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
  std::string ret(demangled_name);
  free(demangled_name);
  return ret;
#elif defined(TI_PLATFORM_WINDOWS)
  PCSTR mangled = mangled_name.c_str();
  char demangled[1024];
  DWORD length =
      UnDecorateSymbolName(mangled, demangled, 1024, UNDNAME_NAME_ONLY);
  return std::string(demangled, size_t(length));
#else
  TI_NOT_IMPLEMENTED
#endif
}

class Demangling : public Task {
  std::string run(const std::vector<std::string> &parameters) override {
    if (parameters.size() == 0) {
      printf("There should be at least one parameter for demangling.\n");
    }
    for (auto p : parameters) {
      printf("Demangled C++ Identifier: %s\n", cpp_demangle(p).c_str());
    }
    return "";
  }
};

TI_IMPLEMENTATION(Task, Demangling, "demangle")

}  // namespace taichi
