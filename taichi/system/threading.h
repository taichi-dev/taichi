/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#if defined(TC_PLATFORM_WINDOWS)
#include <windows.h>
#else
// Mac and Linux
#include <unistd.h>
#endif

TC_NAMESPACE_BEGIN

class PID {
 public:
  static int get_pid() {
#if defined(TC_PLATFORM_WINDOWS)
    return (int)GetCurrentProcessId();
#else
    return (int)getpid();
#endif
  }
  static int get_parent_pid() {
#if defined(TC_PLATFORM_WINDOWS)
    TC_NOT_IMPLEMENTED
    return -1;
#else
    return (int)getppid();
#endif
  }
};

TC_NAMESPACE_END
