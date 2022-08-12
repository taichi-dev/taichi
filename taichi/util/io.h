/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#if defined(TI_PLATFORM_WINDOWS)
#include <filesystem>
#endif

TI_NAMESPACE_BEGIN

inline bool path_exists(const std::string &dir) {
  struct stat buffer;
  return stat(dir.c_str(), &buffer) == 0;
}

// TODO: move to std::filesystem after it's nonexperimental on all platforms
inline void create_directories(const std::string &dir) {
#if defined(TI_PLATFORM_WINDOWS)
  std::filesystem::create_directories(dir);
#else
  int return_code = std::system(fmt::format("mkdir -p {}", dir).c_str());
  if (return_code != 0) {
    throw std::runtime_error(
        fmt::format("Unable to create directory at: {dir}").c_str());
  }
#endif
}

template <typename First, typename... Path>
inline std::string join_path(First &&path, Path &&...others) {
  if constexpr (sizeof...(others) == 0) {
    return std::string(path);
  } else {
    return std::string(path) + "/" +
           taichi::join_path(std::forward<Path>(others)...);
  }
  return "";
}

inline bool remove(const std::string &path) {
  return std::remove(path.c_str()) == 0;
}

template <typename T>
void write_to_disk(const T &dat, std::string fn) {
  FILE *f = fopen(fn.c_str(), "wb");
  fwrite(&dat, sizeof(dat), 1, f);
  fclose(f);
}

template <typename T>
bool read_from_disk(T &dat, std::string fn) {
  FILE *f = fopen(fn.c_str(), "rb");
  if (f == nullptr) {
    return false;
  }
  size_t ret = fread(&dat, sizeof(dat), 1, f);
  if (ret != sizeof(dat)) {
    return false;
  }
  fclose(f);
  return true;
}

template <typename T>
void write_vector_to_disk(std::vector<T> *p_vec, std::string fn) {
  std::vector<T> &vec = *p_vec;
  FILE *f = fopen(fn.c_str(), "wb");
  size_t length = vec.size();
  fwrite(&length, sizeof(length), 1, f);
  fwrite(&vec[0], sizeof(vec[0]), length, f);
  fclose(f);
}

template <typename T>
bool read_vector_from_disk(std::vector<T> *p_vec, std::string fn) {
  std::vector<T> &vec = *p_vec;
  FILE *f = fopen(fn.c_str(), "rb");
  if (f == nullptr) {
    return false;
  }
  size_t length;
  size_t ret = fread(&length, sizeof(length), 1, f);
  if (ret != 1) {
    return false;
  }
  vec.resize(length);
  ret = fread(&vec[0], sizeof(vec[0]), length, f);
  if (ret != length) {
    return false;
  }
  fclose(f);
  return true;
}

TI_NAMESPACE_END
