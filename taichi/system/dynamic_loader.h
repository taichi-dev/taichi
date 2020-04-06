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

// TODO (yuanming-hu): Fix Windows

class DynamicLoader {
 protected:
#ifdef WIN32
  HMODULE dll = nullptr;
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
    TI_ERROR_IF(dlsym_error, "Cannot load function: {}", dlsym_error);
#endif
    TI_ERROR_IF(!func, "Function {} not found", func_name);
    return func;
  }

  template <typename T>
  void load_function(const std::string &func_name, T &f) {
    f = (T)load_function(func_name);
  }

  void close_dll() {
    TI_ERROR_IF(!loaded(), "DLL not opened.");
#ifdef WIN32
    TI_ERROR("Not implemented");
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
