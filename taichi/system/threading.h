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
#if !defined(TC_AMALGAMATED)
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#endif

TC_NAMESPACE_BEGIN

class Spinlock {
 protected:
  std::atomic<bool> latch;

 public:
  Spinlock() : Spinlock(false) {
  }

  Spinlock(bool flag) {
    latch.store(flag);
  }

  Spinlock(int flag) : Spinlock(flag != 0) {
  }

  void lock() {
    bool unlatched = false;
    while (!latch.compare_exchange_weak(unlatched, true,
                                        std::memory_order_acquire)) {
      unlatched = false;
    }
  }

  void unlock() {
    latch.store(false, std::memory_order_release);
  }

  Spinlock(const Spinlock &o) {
    // We just ignore racing condition here...
    latch.store(o.latch.load());
  }

  Spinlock &operator=(const Spinlock &o) {
    // We just ignore racing condition here...
    latch.store(o.latch.load());
    return *this;
  }
};

#if (0)
class ThreadedTaskManager {
 public:
#if !defined(TC_AMALGAMATED)
  class TbbParallelismControl {
    std::unique_ptr<tbb::global_control> c;
   public:
    TbbParallelismControl(int threads) {
      c = std::make_unique<tbb::global_control>(
          tbb::global_control::max_allowed_parallelism, threads);
    }
  };
#endif
  template <typename T>
  void static run(const T &target, int begin, int end, int num_threads) {
#if !defined(TC_AMALGAMATED)
    if (num_threads > 0) {
      tbb::task_arena limited_arena(num_threads);
      limited_arena.execute([&]() { tbb::parallel_for(begin, end, target); });
    } else {
      TC_ASSERT_INFO(
          num_threads == -1,
          fmt::format(
              "num_threads must be a positive number or -1, instead of [{}]",
              num_threads));
      tbb::parallel_for(begin, end, target);
    }
#else
    TC_NOT_IMPLEMENTED
#endif
  }

  template <typename T>
  void static run(const T &target, int end, int num_threads) {
    return run(target, 0, end, num_threads);
  }

  template <typename T>
  void static run(int begin, int end, int num_threads, const T &target) {
    return run(target, begin, end, num_threads);
  }

  template <typename T>
  void static run(int end, int num_threads, const T &target) {
    return run(target, 0, end, num_threads);
  }
};
#endif

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
