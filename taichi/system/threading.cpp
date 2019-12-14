/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <condition_variable>
#include <taichi/system/threading.h>
#include <thread>

TC_NAMESPACE_BEGIN

class ThreadPool {
public:
  std::vector<std::thread> threads;
  std::condition_variable cv;
  std::condition_variable master_cv;
  std::mutex mutex;
  int task_head;
  int task_tail;
  int running_threads;
  uint64 timestamp;
  bool started;
  bool exiting;

  void do_task(int i) {
    double ret = 0.0;
    for (int t = 0; t < 100000000; t++) {
      ret += t * 1e-20;
    }
    TC_P(int(i + ret));
  }

  void target() {
    uint64 last_timestamp = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this, &last_timestamp] {
          return (timestamp > last_timestamp) || this->exiting;
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
          std::unique_lock<std::mutex> lock(mutex);
          task_id = task_head;
          if (task_id >= task_tail)
            break;
          task_head++;
        }
        do_task(task_id);
      }

      {
        std::unique_lock<std::mutex> lock(mutex);
        running_threads--;
        if (running_threads == 0)
          master_cv.notify_one();
      }
    }
  }

  ThreadPool(int num_threads) {
    exiting = false;
    started = false;
    running_threads = 0;
    threads.resize((std::size_t)num_threads);
    for (int i = 0; i < num_threads; i++) {
      threads[i] = std::thread([this] { this->target(); });
    }
  }

  void run(int tail) {
    started = false;
    task_head = 0;
    task_tail = tail;
    timestamp++;
    cv.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex);
      master_cv.wait(lock, [this] { return started && running_threads == 0; });
    }
    TC_ASSERT(task_head == task_tail);
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lg(mutex);
      exiting = true;
    }
    cv.notify_all();
    for (int i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  }
};

bool test_threading() {
  auto tp = ThreadPool(10);
  for (int i = 0; i < 10; i++) {
    tp.run(10);
  }
  return true;
}

TC_NAMESPACE_END
