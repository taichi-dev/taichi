/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <algorithm>
#include <condition_variable>
#include <taichi/system/threading.h>
#include <thread>
#include <vector>
#if defined(TC_PLATFORM_WINDOWS)
#include <windows.h>
#else
// Mac and Linux
#include "threading.h"
#include <unistd.h>

#endif

TC_NAMESPACE_BEGIN

#if defined(min)
#undef min
#endif

bool test_threading() {
  auto tp = ThreadPool();
  for (int j = 0; j < 100; j++) {
    tp.run(10, j + 1, &j, [](void *j, int i) {
      double ret = 0.0;
      for (int t = 0; t < 10000000; t++) {
        ret += t * 1e-20;
      }
      TC_P(int(i + ret + 10 * *(int *)j));
    });
  }
  return true;
}

int PID::get_pid() {
#if defined(TC_PLATFORM_WINDOWS)
  return (int)GetCurrentProcessId();
#else
  return (int)getpid();
#endif
}

int PID::get_parent_pid() {
#if defined(TC_PLATFORM_WINDOWS)
  TC_NOT_IMPLEMENTED
  return -1;
#else
  return (int)getppid();
#endif
}

ThreadPool::ThreadPool() {
  exiting = false;
  started = false;
  running_threads = 0;
  timestamp = 0;
  task_head = 0;
  task_tail = 0;
  max_num_threads = std::thread::hardware_concurrency();
  threads.resize((std::size_t)max_num_threads);
  for (int i = 0; i < max_num_threads; i++) {
    threads[i] = std::thread([this] { this->target(); });
  }
}

void ThreadPool::run(int splits, int desired_num_threads, void *context,
                     CPUTaskFunc *func) {
  this->context = context;
  this->func = func;
  this->desired_num_threads = std::min(desired_num_threads, max_num_threads);
  TC_ASSERT(this->desired_num_threads > 0);
  started = false;
  task_head = 0;
  task_tail = splits;
  timestamp++;

  // wake all slaves
  slave_cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(mutex);
    master_cv.wait(lock, [this] { return started && running_threads == 0; });
  }
  TC_ASSERT(task_head == task_tail);
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
        started = true;
        running_threads++;
      }
    }

    while (true) {
      // For a single parallel task
      int task_id;
      {
        std::lock_guard<std::mutex> lock(mutex);
        task_id = task_head;
        if (task_id >= task_tail)
          break;
        task_head++;
      }
      func(context, task_id);
    }

    {
      std::unique_lock<std::mutex> lock(mutex);
      running_threads--;
      if (running_threads == 0)
        master_cv.notify_one();
    }
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

TC_NAMESPACE_END
