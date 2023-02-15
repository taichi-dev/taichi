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
#else  // POSIX
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

namespace taichi {

inline bool path_exists(const std::string &dir) {
  struct stat buffer;
  return stat(dir.c_str(), &buffer) == 0;
}

// TODO: move to std::filesystem after it's nonexperimental on all platforms
inline void create_directories(const std::string &dir) {
  std::filesystem::create_directories(dir);
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

template <typename Visitor>  // void(const std::string &name, bool is_dir)
inline bool traverse_directory(const std::string &dir, Visitor v) {
#if defined(TI_PLATFORM_WINDOWS)
  namespace fs = std::filesystem;
  std::error_code ec{};
  auto iter = fs::directory_iterator(dir, ec);
  if (ec) {
    return false;
  }
  for (auto &f : iter) {
    v(f.path().filename().string(), f.is_directory());
  }
  return true;
#else  // POSIX
  struct dirent *f = nullptr;
  DIR *directory = ::opendir(dir.c_str());
  if (!directory) {
    return false;
  }
  while ((f = ::readdir(directory))) {
    auto fullpath = join_path(dir, f->d_name);
    struct stat stat_buf;
    auto ret = ::stat(fullpath.c_str(), &stat_buf);
    TI_ASSERT(ret == 0);
    v(f->d_name, S_ISDIR(stat_buf.st_mode));
  }
  auto ret = ::closedir(directory);
  TI_ASSERT(ret == 0);
  return true;
#endif
}

inline std::string filename_extension(const std::string &filename) {
  std::string postfix;
  auto pos = filename.find_last_of('.');
  if (pos != std::string::npos) {
    postfix = filename.substr(pos + 1);
  }
  return postfix;
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

}  // namespace taichi
