/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

    class CacheReadBenchmark : public Benchmark {
    private:
        int working_set_size;
        int workload;
        int n;
        int step;
        // Use float here instead of real to make sure it's 4 bytes
        std::vector<float> data;
    public:
        void initialize(const Config &config) {
            working_set_size = config.get("working_set_size", 1024);
            workload = config.get("workload", 1024);
            step = config.get("step", 1);
            assert_info(working_set_size % 4 == 0, "working_set_size should be a multiple of 4");
            n = working_set_size / 4;
            data.resize(n);
        }
    private:
        void iterate() {
            float a = 0;
            for (unsigned i = 0; i < workload; i++) {
                a += data[(i * step) % n];
            }
            dummy = (int)a;
        }
    };


TC_NAMESPACE_END

