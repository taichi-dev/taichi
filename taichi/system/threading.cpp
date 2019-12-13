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
  bool running;
  bool finished;

  void task() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [this]{ return running;});
      if (finished) {
        break;
      }
      TC_TAG;
    }
  }

  ThreadPool(int num_threads) {
    running = false;
    finished = false;
    threads.resize((std::size_t)num_threads);
    for (int i = 0; i < num_threads; i++) {
      threads[i] = std::thread([this] {
        this->task();
      });
    }
  }

  void start() {
    {
      std::lock_guard<std::mutex> lg(mutex);
      running = true;
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
  auto tp = ThreadPool(4);
  tp.start();
  return true;
}

TC_NAMESPACE_END
