/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

//#define TC_MT_OPENMP

#ifdef TC_MT_OPENMP

#include <omp.h>

#endif

TC_NAMESPACE_BEGIN

class Spinlock {
protected:
    std::atomic<bool> latch;
public:
    Spinlock() : Spinlock(false) {}

    Spinlock(bool flag) {
        latch.store(flag);
    }

    Spinlock(int flag) : Spinlock(flag != 0) {}

    void lock() {
        bool unlatched = false;
        while (!latch.compare_exchange_weak(unlatched, true, std::memory_order_acquire)) {
            unlatched = false;
        }
    }

    void unlock() {
        latch.store(false, std::memory_order_release);
    }

    Spinlock(const Spinlock &o) {
        // We just ignore racing condition here...
        latch.store(o.latch.load());
    }

    Spinlock &operator=(const Spinlock &o) {
        // We just ignore racing condition here...
        latch.store(o.latch.load());
        return *this;
    }
};

class ThreadedTaskManager {
public:
    template <typename T>
    void static run(const T &target, int begin, int end, int num_threads) {
#ifdef TC_MT_OPENMP
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule (dynamic)
        for (int i = begin; i < end; i++) {
            target(i);
        }
#else
        if (num_threads == 1) {
            // Single-threading
            for (int i = begin; i < end; i++) {
                target(i);
            }
        } else {
            // Multi-threading
            std::vector<std::thread *> threads;
            std::vector<int> end_points;
            for (int i = 0; i < num_threads; i++) {
                end_points.push_back(i * (end - begin) / num_threads + begin);
            }
            end_points.push_back(end);
            for (int i = 0; i < num_threads; i++) {
                auto func = [&target, i, &end_points]() {
                    int begin = end_points[i], end = end_points[i + 1];
                    for (int k = begin; k < end; k++) {
                        target(k);
                    }
                };
                threads.push_back(new std::thread(func));
            }
            for (int i = 0; i < num_threads; i++) {
                threads[i]->join();
                delete threads[i];
            }
        }
#endif
    }

    template <typename T>
    void static run(const T &target, int end, int num_threads) {
        return run(target, 0, end, num_threads);
    }

    template <typename T>
    void static run(int begin, int end, int num_threads, const T &target) {
        return run(target, begin, end, num_threads);
    }

    template <typename T>
    void static run(int end, int num_threads, const T &target) {
        return run(target, 0, end, num_threads);
    }
};

TC_NAMESPACE_END