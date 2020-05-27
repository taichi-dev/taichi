/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include "taichi/common/core.h"
#include <thread>

TI_NAMESPACE_BEGIN

using RangeForTaskFunc = void(void *, int i);
using ParallelFor = void(int n, int num_threads, void *, RangeForTaskFunc func);

class PID {
 public:
  static int get_pid();
  static int get_parent_pid();
};

class ThreadPool {
 public:
  std::vector<std::thread> threads;
  std::condition_variable slave_cv;
  std::condition_variable master_cv;
  std::mutex mutex;
  std::atomic<int> task_head;
  int task_tail;
  int running_threads;
  int max_num_threads;
  int desired_num_threads;
  uint64 timestamp;
  uint64 last_finished;
  bool started;
  bool exiting;
  RangeForTaskFunc *func;
  void *context;
  int thread_counter;

  ThreadPool();

  void run(int splits,
           int desired_num_threads,
           void *context,
           RangeForTaskFunc *func);

  static void static_run(ThreadPool *pool,
                         int splits,
                         int desired_num_threads,
                         void *context,
                         RangeForTaskFunc *func) {
    return pool->run(splits, desired_num_threads, context, func);
  }

  void target();

  ~ThreadPool();
};

TI_NAMESPACE_END
