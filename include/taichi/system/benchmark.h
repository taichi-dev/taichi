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
        virtual void setup() {};
        virtual void iterate() {};
        virtual void finalize() {};
    public:
        virtual void initialize(const Config &config) {
            warm_up_iterations = config.get("warm_up_iterations", 16);
        }
        virtual real run(int iterations=16) {
            setup();
            for (int i = 0; i < warm_up_iterations; i++) {
                iterate();
            }
            double t = Time::get_time();
            for (int i = 0; i < warm_up_iterations; i++) {
                iterate();
            }
            real elapsed = (real)(Time::get_time() - t);
            finalize();
            return elapsed / iterations;
        }
    };


TC_NAMESPACE_END
