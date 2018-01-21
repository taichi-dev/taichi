/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN

class UnitDLL {
 protected:
#ifdef WIN32
  HINSTANCE dll = nullptr;
#else
  void *dll = nullptr;
#endif

  typedef int (*Func)(void);

 public:
  void load_dll(const std::string dll_path) {
#ifdef WIN32
    dll = LoadLibrary(dll_path.c_str());
#else
    dll = dlopen(dll_path.c_str(), RTLD_LAZY);
#endif
    if (!dll) {
      TC_ERROR(std::string("Cannot load library: " + dll_path));
    }
  }

  Func load_function(const std::string &func_name) {
#ifdef WIN32
    Func func = (Func)GetProcAddress(dll, func_name.c_str());
#else
    auto func = (Func)dlsym(dll, func_name.c_str());
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      TC_ERROR(std::string("Cannot load function: ") + dlsym_error);
    }
#endif
    assert_info(func != nullptr, "Function " + func_name + " not found");
    return func;
  }

  UnitDLL() {
  }

  UnitDLL(const std::string &dll_path) {
    open_dll(dll_path);
  }

  void open_dll(const std::string &dll_path) {
    assert(dll == nullptr);
    load_dll(dll_path);
    auto loader = load_function("on_load");
    on_load = [loader]() { loader(); };
    on_load();
    auto unloader = load_function("on_unload");
    on_unload = [unloader]() { unloader(); };
  }

  void close_dll() {
    assert_info(loaded(), "Dll not opened.");
    on_unload();
#ifdef WIN32
    TC_P("Not implemented");
#else
    dlclose(dll);
#endif
    dll = nullptr;
  }

  bool loaded() const {
    return dll != nullptr;
  }

  ~UnitDLL() {
    if (loaded())
      close_dll();
  }

 protected:
  std::function<void(void)> on_load;
  std::function<void(void)> on_unload;
};

TC_NAMESPACE_END
