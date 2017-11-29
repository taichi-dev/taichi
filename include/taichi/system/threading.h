/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#ifdef __WIN32__

#else
// Mac and Linux
#include <unistd.h>
#include <tbb/tbb.h>
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

class ThreadedTaskManager {
 public:
  template <typename T>
  void static run(const T &target, int begin, int end, int num_threads) {
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

class PID {
 public:
  static int get_pid() {
    return (int)getpid();
  }
  static int get_parent_pid() {
    return (int)getppid();
  }
};

TC_NAMESPACE_END