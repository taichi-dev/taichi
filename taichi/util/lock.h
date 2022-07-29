#pragma once

#include "taichi/common/core.h"
#include <thread>

#if defined(TI_PLATFORM_WINDOWS)
#include <io.h>
#include <fcntl.h>
#else  // POSIX
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace taichi {

inline bool try_lock_with_file(const std::string &path) {
  int fd{-1};
#if defined(TI_PLATFORM_WINDOWS)
  // See
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/sopen-s-wsopen-s
  ::_sopen_s(&fd, path.c_str(), _O_CREAT | _O_EXCL, _SH_DENYNO,
             _S_IREAD | _S_IWRITE);
  if (fd != -1)
    ::_close(fd);
#else
  // See https://www.man7.org/linux/man-pages/man2/open.2.html
  fd = ::open(path.c_str(), O_CREAT | O_EXCL,
              S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
  if (fd != -1)
    ::close(fd);
#endif
  return fd != -1;
}

inline bool unlock_with_file(const std::string &path) {
  return std::remove(path.c_str()) == 0;
}

inline bool lock_with_file(const std::string &path,
                           int ms_delay = 50,
                           int try_count = 5) {
  if (try_lock_with_file(path)) {
    return true;
  }
  for (int i = 1; i < try_count; ++i) {
    std::chrono::milliseconds delay{ms_delay};
    std::this_thread::sleep_for(delay);
    if (try_lock_with_file(path)) {
      return true;
    }
  }
  return false;
}

}  // namespace taichi
