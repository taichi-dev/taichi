#include "taichi/common/dynamic_loader.h"

#ifdef WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <dlfcn.h>
#endif

namespace taichi {

bool DynamicLoader::check_lib_loaded(const std::string &lib_path) {
  bool loaded = false;
#ifdef WIN32
  loaded = ((HMODULE)GetModuleHandleA(lib_path.c_str()) != nullptr);
#else
  loaded = (dlopen(lib_path.c_str(), RTLD_NOLOAD) != nullptr);
#endif
  return loaded;
}

DynamicLoader::DynamicLoader(const std::string &dll_path) {
  load_dll(dll_path);
}

void DynamicLoader::load_dll(const std::string &dll_path) {
#ifdef WIN32
  dll_ = (HMODULE)LoadLibraryA(dll_path.c_str());
#else
  dll_ = dlopen(dll_path.c_str(), RTLD_LAZY);
#endif
}

void *DynamicLoader::load_function(const std::string &func_name) {
  TI_ASSERT_INFO(loaded(), "DLL not opened");
#ifdef WIN32
  auto func = (void *)GetProcAddress((HMODULE)dll_, func_name.c_str());
#else
  auto func = dlsym(dll_, func_name.c_str());
  const char *dlsym_error = dlerror();
  TI_ERROR_IF(dlsym_error, "Cannot load function: {}", dlsym_error);
#endif
  TI_ERROR_IF(!func, "Function {} not found", func_name);
  return func;
}

void DynamicLoader::close_dll() {
  TI_ASSERT_INFO(loaded(), "DLL not opened");
#ifdef WIN32
  FreeLibrary((HMODULE)dll_);
#else
  dlclose(dll_);
#endif
  dll_ = nullptr;
}

DynamicLoader::~DynamicLoader() {
  if (loaded())
    close_dll();
}

bool DynamicLoader::loaded() const {
  return dll_ != nullptr;
}

}  // namespace taichi
