#include "taichi/system/dynamic_loader.h"

#ifdef WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <dlfcn.h>
#endif

TI_NAMESPACE_BEGIN

DynamicLoader::DynamicLoader(const std::string &dll_path) {
  load_dll(dll_path);
}

void DynamicLoader::load_dll(const std::string &dll_path) {
#ifdef WIN32
  dll = (HMODULE)LoadLibraryA(dll_path.c_str());
#else
  dll = dlopen(dll_path.c_str(), RTLD_LAZY);
#endif
}

void *DynamicLoader::load_function(const std::string &func_name) {
  TI_ASSERT(loaded());
#ifdef WIN32
  auto func = (void *)GetProcAddress((HMODULE)dll, func_name.c_str());
#else
  auto func = dlsym(dll, func_name.c_str());
  const char *dlsym_error = dlerror();
  TI_ERROR_IF(dlsym_error, "Cannot load function: {}", dlsym_error);
#endif
  TI_ERROR_IF(!func, "Function {} not found", func_name);
  return func;
}

void DynamicLoader::close_dll() {
  TI_ERROR_IF(!loaded(), "DLL not opened.");
#ifdef WIN32
  FreeLibrary((HMODULE)dll);
#else
  dlclose(dll);
#endif
  dll = nullptr;
}

DynamicLoader::~DynamicLoader() {
  if (loaded())
    close_dll();
}

bool DynamicLoader::loaded() const {
  return dll != nullptr;
}

TI_NAMESPACE_END
