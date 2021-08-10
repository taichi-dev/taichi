// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS
#include "main.h"
#include <Eigen/CXX11/ThreadPool>

// Visual studio doesn't implement a rand_r() function since its
// implementation of rand() is already thread safe
int rand_reentrant(unsigned int* s) {
#ifdef EIGEN_COMP_MSVC_STRICT
  EIGEN_UNUSED_VARIABLE(s);
  return rand();
#else
  return rand_r(s);
#endif
}

static void test_basic_eventcount()
{
  MaxSizeVector<EventCount::Waiter> waiters(1);
  waiters.resize(1);
  EventCount ec(waiters);
  EventCount::Waiter& w = waiters[0];
  ec.Notify(false);
  ec.Prewait(&w);
  ec.Notify(true);
  ec.CommitWait(&w);
  ec.Prewait(&w);
  ec.CancelWait(&w);
}

// Fake bounded counter-based queue.
struct TestQueue {
  std::atomic<int> val_;
  static const int kQueueSize = 10;

  TestQueue() : val_() {}

  ~TestQueue() { VERIFY_IS_EQUAL(val_.load(), 0); }

  bool Push() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      VERIFY_GE(val, 0);
      VERIFY_LE(val, kQueueSize);
      if (val == kQueueSize) return false;
      if (val_.compare_exchange_weak(val, val + 1, std::memory_order_relaxed))
        return true;
    }
  }

  bool Pop() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      VERIFY_GE(val, 0);
      VERIFY_LE(val, kQueueSize);
      if (val == 0) return false;
      if (val_.compare_exchange_weak(val, val - 1, std::memory_order_relaxed))
        return true;
    }
  }

  bool Empty() { return val_.load(std::memory_order_relaxed) == 0; }
};

const int TestQueue::kQueueSize;

// A number of producers send messages to a set of consumers using a set of
// fake queues. Ensure that it does not crash, consumers don't deadlock and
// number of blocked and unblocked threads match.
static void test_stress_eventcount()
{
  const int kThreads = std::thread::hardware_concurrency();
  static const int kEvents = 1 << 16;
  static const int kQueues = 10;

  MaxSizeVector<EventCount::Waiter> waiters(kThreads);
  waiters.resize(kThreads);
  EventCount ec(waiters);
  TestQueue queues[kQueues];

  std::vector<std::unique_ptr<std::thread>> producers;
  for (int i = 0; i < kThreads; i++) {
    producers.emplace_back(new std::thread([&ec, &queues]() {
      unsigned int rnd = static_cast<unsigned int>(std::hash<std::thread::id>()(std::this_thread::get_id()));
      for (int j = 0; j < kEvents; j++) {
        unsigned idx = rand_reentrant(&rnd) % kQueues;
        if (queues[idx].Push()) {
          ec.Notify(false);
          continue;
        }
        EIGEN_THREAD_YIELD();
        j--;
      }
    }));
  }

  std::vector<std::unique_ptr<std::thread>> consumers;
  for (int i = 0; i < kThreads; i++) {
    consumers.emplace_back(new std::thread([&ec, &queues, &waiters, i]() {
      EventCount::Waiter& w = waiters[i];
      unsigned int rnd = static_cast<unsigned int>(std::hash<std::thread::id>()(std::this_thread::get_id()));
      for (int j = 0; j < kEvents; j++) {
        unsigned idx = rand_reentrant(&rnd) % kQueues;
        if (queues[idx].Pop()) continue;
        j--;
        ec.Prewait(&w);
        bool empty = true;
        for (int q = 0; q < kQueues; q++) {
          if (!queues[q].Empty()) {
            empty = false;
            break;
          }
        }
        if (!empty) {
          ec.CancelWait(&w);
          continue;
        }
        ec.CommitWait(&w);
      }
    }));
  }

  for (int i = 0; i < kThreads; i++) {
    producers[i]->join();
    consumers[i]->join();
  }
}

void test_cxx11_eventcount()
{
  CALL_SUBTEST(test_basic_eventcount());
  CALL_SUBTEST(test_stress_eventcount());
}
