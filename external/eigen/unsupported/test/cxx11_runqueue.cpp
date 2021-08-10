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
#include <cstdlib>
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

void test_basic_runqueue()
{
  RunQueue<int, 4> q;
  // Check empty state.
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PopFront());
  std::vector<int> stolen;
  VERIFY_IS_EQUAL(0u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(0u, stolen.size());
  // Push one front, pop one front.
  VERIFY_IS_EQUAL(0, q.PushFront(1));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(1, q.PopFront());
  VERIFY_IS_EQUAL(0u, q.Size());
  // Push front to overflow.
  VERIFY_IS_EQUAL(0, q.PushFront(2));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(3));
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(4));
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(5));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(6, q.PushFront(6));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(5, q.PopFront());
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(4, q.PopFront());
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(3, q.PopFront());
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(2, q.PopFront());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PopFront());
  // Push one back, pop one back.
  VERIFY_IS_EQUAL(0, q.PushBack(7));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(7, stolen[0]);
  VERIFY_IS_EQUAL(0u, q.Size());
  stolen.clear();
  // Push back to overflow.
  VERIFY_IS_EQUAL(0, q.PushBack(8));
  VERIFY_IS_EQUAL(1u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(9));
  VERIFY_IS_EQUAL(2u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(10));
  VERIFY_IS_EQUAL(3u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushBack(11));
  VERIFY_IS_EQUAL(4u, q.Size());
  VERIFY_IS_EQUAL(12, q.PushBack(12));
  VERIFY_IS_EQUAL(4u, q.Size());
  // Pop back in halves.
  VERIFY_IS_EQUAL(2u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(2u, stolen.size());
  VERIFY_IS_EQUAL(10, stolen[0]);
  VERIFY_IS_EQUAL(11, stolen[1]);
  VERIFY_IS_EQUAL(2u, q.Size());
  stolen.clear();
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(9, stolen[0]);
  VERIFY_IS_EQUAL(1u, q.Size());
  stolen.clear();
  VERIFY_IS_EQUAL(1u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(1u, stolen.size());
  VERIFY_IS_EQUAL(8, stolen[0]);
  stolen.clear();
  VERIFY_IS_EQUAL(0u, q.PopBackHalf(&stolen));
  VERIFY_IS_EQUAL(0u, stolen.size());
  // Empty again.
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
  VERIFY_IS_EQUAL(0, q.PushFront(1));
  VERIFY_IS_EQUAL(0, q.PushFront(2));
  VERIFY_IS_EQUAL(0, q.PushFront(3));
  VERIFY_IS_EQUAL(1, q.PopBack());
  VERIFY_IS_EQUAL(2, q.PopBack());
  VERIFY_IS_EQUAL(3, q.PopBack());
  VERIFY(q.Empty());
  VERIFY_IS_EQUAL(0u, q.Size());
}

// Empty tests that the queue is not claimed to be empty when is is in fact not.
// Emptiness property is crucial part of thread pool blocking scheme,
// so we go to great effort to ensure this property. We create a queue with
// 1 element and then push 1 element (either front or back at random) and pop
// 1 element (either front or back at random). So queue always contains at least
// 1 element, but otherwise changes chaotically. Another thread constantly tests
// that the queue is not claimed to be empty.
void test_empty_runqueue()
{
  RunQueue<int, 4> q;
  q.PushFront(1);
  std::atomic<bool> done(false);
  std::thread mutator([&q, &done]() {
    unsigned rnd = 0;
    std::vector<int> stolen;
    for (int i = 0; i < 1 << 18; i++) {
      if (rand_reentrant(&rnd) % 2)
        VERIFY_IS_EQUAL(0, q.PushFront(1));
      else
        VERIFY_IS_EQUAL(0, q.PushBack(1));
      if (rand_reentrant(&rnd) % 2)
        VERIFY_IS_EQUAL(1, q.PopFront());
      else {
        for (;;) {
          if (q.PopBackHalf(&stolen) == 1) {
            stolen.clear();
            break;
          }
          VERIFY_IS_EQUAL(0u, stolen.size());
        }
      }
    }
    done = true;
  });
  while (!done) {
    VERIFY(!q.Empty());
    int size = q.Size();
    VERIFY_GE(size, 1);
    VERIFY_LE(size, 2);
  }
  VERIFY_IS_EQUAL(1, q.PopFront());
  mutator.join();
}

// Stress is a chaotic random test.
// One thread (owner) calls PushFront/PopFront, other threads call PushBack/
// PopBack. Ensure that we don't crash, deadlock, and all sanity checks pass.
void test_stress_runqueue()
{
  static const int kEvents = 1 << 18;
  RunQueue<int, 8> q;
  std::atomic<int> total(0);
  std::vector<std::unique_ptr<std::thread>> threads;
  threads.emplace_back(new std::thread([&q, &total]() {
    int sum = 0;
    int pushed = 1;
    int popped = 1;
    while (pushed < kEvents || popped < kEvents) {
      if (pushed < kEvents) {
        if (q.PushFront(pushed) == 0) {
          sum += pushed;
          pushed++;
        }
      }
      if (popped < kEvents) {
        int v = q.PopFront();
        if (v != 0) {
          sum -= v;
          popped++;
        }
      }
    }
    total += sum;
  }));
  for (int i = 0; i < 2; i++) {
    threads.emplace_back(new std::thread([&q, &total]() {
      int sum = 0;
      for (int j = 1; j < kEvents; j++) {
        if (q.PushBack(j) == 0) {
          sum += j;
          continue;
        }
        EIGEN_THREAD_YIELD();
        j--;
      }
      total += sum;
    }));
    threads.emplace_back(new std::thread([&q, &total]() {
      int sum = 0;
      std::vector<int> stolen;
      for (int j = 1; j < kEvents;) {
        if (q.PopBackHalf(&stolen) == 0) {
          EIGEN_THREAD_YIELD();
          continue;
        }
        while (stolen.size() && j < kEvents) {
          int v = stolen.back();
          stolen.pop_back();
          VERIFY_IS_NOT_EQUAL(v, 0);
          sum += v;
          j++;
        }
      }
      while (stolen.size()) {
        int v = stolen.back();
        stolen.pop_back();
        VERIFY_IS_NOT_EQUAL(v, 0);
        while ((v = q.PushBack(v)) != 0) EIGEN_THREAD_YIELD();
      }
      total -= sum;
    }));
  }
  for (size_t i = 0; i < threads.size(); i++) threads[i]->join();
  VERIFY(q.Empty());
  VERIFY(total.load() == 0);
}

void test_cxx11_runqueue()
{
  CALL_SUBTEST_1(test_basic_runqueue());
  CALL_SUBTEST_2(test_empty_runqueue());
  CALL_SUBTEST_3(test_stress_runqueue());
}
