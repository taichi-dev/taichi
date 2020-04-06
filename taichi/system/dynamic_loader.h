/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#ifdef WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <dlfcn.h>
#endif

#include "taichi/common/util.h"

TI_NAMESPACE_BEGIN

class DynamicLoader {
 protected:
#ifdef WIN32
  HINSTANCE dll = nullptr;
#else
  void *dll = nullptr;
#endif

 private:
  void load_dll(const std::string &dll_path) {
#ifdef WIN32
    TI_NOT_IMPLEMENTED
    // dll = LoadLibrary(dll_path.c_str());
#else
    dll = dlopen(dll_path.c_str(), RTLD_LAZY);
#endif
    if (!dll) {
      TI_ERROR(std::string("Cannot load library: " + dll_path));
    }
  }

 public:
  DynamicLoader(const std::string &dll_path) {
    load_dll(dll_path);
  }

  void *load_function(const std::string &func_name) {
#ifdef WIN32
    auto func = (void *)GetProcAddress(dll, func_name.c_str());
#else
    auto func = dlsym(dll, func_name.c_str());
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      TI_ERROR(std::string("Cannot load function: ") + dlsym_error);
    }
#endif
    assert_info(func != nullptr, "Function " + func_name + " not found");
    return func;
  }

  template <typename T>
  void load_function(const std::string &func_name, T &f) {
    f = load_function<T>(func_name);
  }

  void close_dll() {
    assert_info(loaded(), "Dll not opened.");
#ifdef WIN32
    TI_P("Not implemented");
#else
    dlclose(dll);
#endif
    dll = nullptr;
  }

  bool loaded() const {
    return dll != nullptr;
  }

  ~DynamicLoader() {
    if (loaded())
      close_dll();
  }
};

TI_NAMESPACE_END
