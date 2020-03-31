/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <algorithm>
#include <condition_variable>
#include "taichi/system/threading.h"
#include <thread>
#include <vector>
#if defined(TI_PLATFORM_WINDOWS)
#include "taichi/platform/windows/windows.h"
#else
// Mac and Linux
#include "threading.h"
#include <unistd.h>

#endif

TI_NAMESPACE_BEGIN

bool test_threading() {
  auto tp = ThreadPool();
  for (int j = 0; j < 100; j++) {
    tp.run(10, j + 1, &j, [](void *j, int i) {
      double ret = 0.0;
      for (int t = 0; t < 10000000; t++) {
        ret += t * 1e-20;
      }
      TI_P(int(i + ret + 10 * *(int *)j));
    });
  }
  return true;
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

ThreadPool::ThreadPool() {
  exiting = false;
  started = false;
  running_threads = 0;
  timestamp = 1;
  last_finished = 0;
  task_head = 0;
  task_tail = 0;
  thread_counter = 0;
  max_num_threads = std::thread::hardware_concurrency();
  threads.resize((std::size_t)max_num_threads);
  for (int i = 0; i < max_num_threads; i++) {
    threads[i] = std::thread([this] { this->target(); });
  }
}

void ThreadPool::run(int splits,
                     int desired_num_threads,
                     void *context,
                     RangeForTaskFunc *func) {
  {
    std::lock_guard _(mutex);
    this->context = context;
    this->func = func;
    this->desired_num_threads = std::min(desired_num_threads, max_num_threads);
    TI_ASSERT(this->desired_num_threads > 0);
    // TI_P(this->desired_num_threads);
    started = false;
    task_head = 0;
    task_tail = splits;
    timestamp++;
    TI_ASSERT(timestamp < (1LL << 62));  // avoid overflowing here
  }

  // wake up all slaves
  slave_cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(mutex);
    // TODO: the workers may have finished before master waiting on master_cv
    master_cv.wait(lock, [this] { return started && running_threads == 0; });
  }
  TI_ASSERT(task_head >= task_tail);
}

void ThreadPool::target() {
  uint64 last_timestamp = 0;
  int thread_id;
  {
    std::lock_guard<std::mutex> lock(mutex);
    thread_id = thread_counter++;
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      slave_cv.wait(lock, [this, last_timestamp, thread_id] {
        return (timestamp > last_timestamp &&
                thread_id < desired_num_threads) ||
               this->exiting;
      });
      last_timestamp = timestamp;
      if (exiting) {
        break;
      } else {
        if (last_finished >= last_timestamp) {
          continue;
          // This could happen when part of the desired threads wake up and
          // finish all the task, and then this thread wake up finding nothing
          // to do. Should skip this task directly.
        } else {
          started = true;
          running_threads++;
        }
      }
    }

    while (true) {
      // For a single parallel task
      int task_id;
      {
        task_id = task_head.fetch_add(1, std::memory_order_relaxed);
        if (task_id >= task_tail)
          break;
      }
      func(context, task_id);
    }

    bool all_finished = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      running_threads--;
      if (running_threads == 0) {
        all_finished = true;
        last_finished = last_timestamp;
      }
    }
    if (all_finished)
      master_cv.notify_one();
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lg(mutex);
    exiting = true;
  }
  slave_cv.notify_all();
  for (auto &th : threads)
    th.join();
}

TI_NAMESPACE_END
