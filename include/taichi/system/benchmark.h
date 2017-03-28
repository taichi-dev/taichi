/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>
#include <taichi/system/timer.h>

TC_NAMESPACE_BEGIN

    class Benchmark : public Unit {
    protected:
        int dummy;
        int warm_up_iterations;
        int workload;
        virtual void setup() {};
        virtual void iterate() = 0;
        virtual void finalize() {};
    public:
        virtual void initialize(const Config &config) {
            warm_up_iterations = config.get("warm_up_iterations", 16);
            workload = config.get("workload", 1024);
        }
        virtual real run(int iterations=16) {
            setup();
            for (int i = 0; i < warm_up_iterations; i++) {
                iterate();
            }
            double t = Time::get_time();
            for (int i = 0; i < iterations; i++) {
                iterate();
            }
            real elapsed = (real)(Time::get_time() - t);
            finalize();
            return elapsed / (iterations * workload);
        }
    };
    TC_INTERFACE(Benchmark)


TC_NAMESPACE_END
