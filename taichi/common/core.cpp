/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/core.h"

#include "taichi/common/version.h"
#include "taichi/common/commit_hash.h"

#if defined(TI_PLATFORM_WINDOWS)
#include "taichi/platform/windows/windows.h"
#else
// Mac and Linux
#include <unistd.h>
#endif

TI_NAMESPACE_BEGIN

extern "C" {
#if defined(TI_PLATFORM_LINUX) && defined(TI_ARCH_x64)
// Avoid dependency on glibc 2.27
// log2f is used by a third party .a file, so we have to define a wrapper.
// https://stackoverflow.com/questions/8823267/linking-against-older-symbol-version-in-a-so-file
__asm__(".symver log2f,log2f@GLIBC_2.2.5");
float __wrap_log2f(float x) {
  return log2f(x);
}
#endif
}

std::string python_package_dir;

std::string get_python_package_dir() {
  return python_package_dir;
}

void set_python_package_dir(const std::string &dir) {
  python_package_dir = dir;
}

std::string get_repo_dir() {
  // release mode. Use ~/.taichi as root
#if defined(TI_PLATFORM_WINDOWS)
  return "C:/taichi_cache/";
#else
  auto home = std::getenv("HOME");
  TI_ASSERT(home != nullptr);
  return std::string(home) + "/.taichi/";
#endif
}

CoreState &CoreState::get_instance() {
  static CoreState state;
  return state;
}

int __trash__;

std::string get_version_string() {
  return fmt::format("{}.{}.{}", get_version_major(), get_version_minor(),
                     get_version_patch());
}

int get_version_major() {
  return std::atoi(TI_VERSION_MAJOR);
}

int get_version_minor() {
  return std::atoi(TI_VERSION_MINOR);
}

int get_version_patch() {
  return std::atoi(TI_VERSION_PATCH);
}

std::string get_commit_hash() {
  return TI_COMMIT_HASH;
}

std::string get_cuda_version_string() {
  return TI_CUDAVERSION;
}

int PID::get_pid() {
#if defined(TI_PLATFORM_WINDOWS)
  return (int)GetCurrentProcessId();
#else
  return (int)getpid();
#endif
}

int PID::get_parent_pid() {
#if defined(TI_PLATFORM_WINDOWS)
  TI_NOT_IMPLEMENTED
  return -1;
#else
  return (int)getppid();
#endif
}

TI_NAMESPACE_END
