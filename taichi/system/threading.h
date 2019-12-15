/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <taichi/common/util.h>
#include <thread>

TC_NAMESPACE_BEGIN

using CPUTaskFunc = void(void *, int i);
using ParallelFor = void(int n, int num_threads, void *, CPUTaskFunc func);

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
  int task_head;
  int task_tail;
  int running_threads;
  int max_num_threads;
  int desired_num_threads;
  uint64 timestamp;
  bool started;
  bool exiting;
  CPUTaskFunc *func;
  void *context;
  int thread_counter;
  std::function<ParallelFor> parallel_for_func;

  ThreadPool();

  void run(int splits, int desired_num_threads, void *context,
           CPUTaskFunc *func);

  void target();

  ParallelFor *get_parallel_for();

  ~ThreadPool();
};

TC_NAMESPACE_END
