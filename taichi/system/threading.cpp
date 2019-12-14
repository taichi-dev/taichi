/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/system/threading.h>
#include <thread>
#include <condition_variable>

TC_NAMESPACE_BEGIN

class ThreadPool {
public:
  std::vector<std::thread> threads;
  std::condition_variable cv;
  std::mutex mutex;
  std::atomic<int> task_head;
  int task_tail;
  bool running;
  bool finished;


  void do_task(int i) {
    double ret = 0.0;
    for (int t = 0; t < 100000000; t++) {
      ret += t * 1e-20;
    }
    TC_P(i + ret);
  }

  void target() {
    while (true) {
      int task_id;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]{ return running;});
        task_id = task_head++;
        if (task_id > task_tail) {
          break;
        }
      }
      do_task(task_id);
    }
  }

  ThreadPool(int num_threads) {
    running = false;
    finished = false;
    threads.resize((std::size_t)num_threads);
    for (int i = 0; i < num_threads; i++) {
      threads[i] = std::thread([this] {
        this->target();
      });
    }
  }

  void start(int tail) {
    {
      std::lock_guard<std::mutex> lg(mutex);
      running = true;
      task_head = 0;
      task_tail = tail;
    }
    cv.notify_all();
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lg(mutex);
      finished = true;
    }
    cv.notify_all();
    for (int i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  }
};

bool test_threading() {
  auto tp = ThreadPool(10);
  tp.start(1000);
  return true;
}

TC_NAMESPACE_END
