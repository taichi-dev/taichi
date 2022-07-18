#pragma once

#include <string>

#ifdef WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <dlfcn.h>
#endif

namespace capi {

inline bool is_cuda_available() {
  void *dll = nullptr;

#ifdef WIN32
  std::string dll_path = "nvcuda.dll";
  dll = (HMODULE)LoadLibraryA(dll_path.c_str());
#else
  std::string dll_path = "libcuda.so";
  dll = dlopen(dll_path.c_str(), RTLD_LAZY);
#endif
  if (dll)
    return true;

  return false;
}

}  // namespace capi
