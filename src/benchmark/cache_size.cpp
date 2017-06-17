/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

class CacheStridedReadBenchmark : public Benchmark {
private:
    int working_set_size;
    int n;
    int step;
    // Use float here instead of real to make sure it's 4 bytes
    std::vector<int> data;
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        working_set_size = config.get("working_set_size", 1024);
        step = config.get("step", 1);
        assert_info(working_set_size % 4 == 0, "working_set_size should be a multiple of 4");
        assert_info((working_set_size & (working_set_size - 1)) == 0, "working_set_size should be a power of 2");
        n = working_set_size / 4;
        data.resize(n);
        for (auto &d : data) {
            d = 1;
        }
    }

protected:
    void iterate() override {
        int a0 = 0;
        int a1 = 0;
        int a2 = 0;
        int a3 = 0;
        unsigned int j = 0;
        for (unsigned i = 0; i < workload / 4; i++) {
            j = (j + step) & (n - 1);
            a0 += data[j];
            j = (j + step) & (n - 1);
            a1 += data[j];
            j = (j + step) & (n - 1);
            a2 += data[j];
            j = (j + step) & (n - 1);
            a3 += data[j];
        }
        dummy = (int) (a0 + a1 + a2 + a3);
    }
};

TC_IMPLEMENTATION(Benchmark, CacheStridedReadBenchmark, "cache_strided_read");

TC_NAMESPACE_END

