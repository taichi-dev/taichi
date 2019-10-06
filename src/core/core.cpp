/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

bool is_release() {
  auto dir = std::getenv("TAICHI_REPO_DIR");
  if (dir != nullptr || std::string(dir) == "") {
    return true;
  } else {
    return false;
  }
}

std::string python_package_dir;

std::string get_python_package_dir() {
  return python_package_dir;
}

std::string set_python_package_dir(const std::string &dir) {
  python_package_dir = dir;
}

std::string get_repo_dir() {
  auto dir = std::getenv("TAICHI_REPO_DIR");
  if (is_release()) {
    // release mode. Use ~/.taichi as root
    auto home = std::getenv("HOME");
    TC_ASSERT(home != nullptr);
    return std::string(home) + "/.taichi/";
  } else {
    return std::string(dir);
  }
}

CoreState &CoreState::get_instance() {
  static CoreState state;
  return state;
}

int __trash__;

TC_NAMESPACE_END
