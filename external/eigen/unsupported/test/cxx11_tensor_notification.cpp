// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Vijay Vasudevan <vrv@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS

#include <stdlib.h>
#include "main.h"
#include <Eigen/CXX11/Tensor>

#if EIGEN_OS_WIN || EIGEN_OS_WIN64
#include <windows.h>
void sleep(int seconds) {
  Sleep(seconds*1000);
}
#else
#include <unistd.h>
#endif


namespace {

void WaitAndAdd(Eigen::Notification* n, int* counter) {
  n->Wait();
  *counter = *counter + 1;
}

}  // namespace

static void test_notification_single()
{
  ThreadPool thread_pool(1);

  int counter = 0;
  Eigen::Notification n;
  std::function<void()> func = std::bind(&WaitAndAdd, &n, &counter);
  thread_pool.Schedule(func);
  sleep(1);

  // The thread should be waiting for the notification.
  VERIFY_IS_EQUAL(counter, 0);

  // Unblock the thread
  n.Notify();

  sleep(1);

  // Verify the counter has been incremented
  VERIFY_IS_EQUAL(counter, 1);
}

// Like test_notification_single() but enqueues multiple threads to
// validate that all threads get notified by Notify().
static void test_notification_multiple()
{
  ThreadPool thread_pool(1);

  int counter = 0;
  Eigen::Notification n;
  std::function<void()> func = std::bind(&WaitAndAdd, &n, &counter);
  thread_pool.Schedule(func);
  thread_pool.Schedule(func);
  thread_pool.Schedule(func);
  thread_pool.Schedule(func);
  sleep(1);
  VERIFY_IS_EQUAL(counter, 0);
  n.Notify();
  sleep(1);
  VERIFY_IS_EQUAL(counter, 4);
}

void test_cxx11_tensor_notification()
{
  CALL_SUBTEST(test_notification_single());
  CALL_SUBTEST(test_notification_multiple());
}
