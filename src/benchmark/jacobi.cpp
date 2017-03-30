/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

template<typename T>
T get_initial_entry(int i, int j, int k) {
    return (unsigned(i * 10000000007 + j * 321343212 + k * 412344739)) * 1e-10;
}

template<typename T>
class JacobiSerial;

template<typename T>
class JacobiBruteForce : public Benchmark {
private:
    int n;
    std::vector<std::vector<std::vector<T>>> data[2];
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        n = config.get_int("n");
        assert_info((n & (n - 1)) == 0, "n should be a power of 2");
        workload = n * n * n;
        for (int l = 0; l < 2; l++) {
            data[l].resize(n);
            for (int i = 0; i < n; i++) {
                data[l][i].resize(n);
                for (int j = 0; j < n; j++) {
                    data[l][i][j].resize(n);
                }
            }
        }
    }

protected:

    virtual void setup() override {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    data[0][i][j][k] = get_initial_entry<T>(i, j, k);
                }
            }
        }
    }

    void iterate() override {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    T t(0);
                    if (i > 0)
                        t += data[0][i - 1][j][k];
                    if (j > 0)
                        t += data[0][i][j - 1][k];
                    if (k > 0)
                        t += data[0][i][j][k - 1];
                    if (i + 1 < n)
                        t += data[0][i + 1][j][k];
                    if (j + 1 < n)
                        t += data[0][i][j + 1][k];
                    if (k + 1 < n)
                        t += data[0][i][j][k + 1];
                    data[1][i][j][k] = t / T(6.0);
                }
            }
        }
        std::swap(data[0], data[1]);
    }

    friend JacobiSerial<T>;
};

template<typename T>
class JacobiSerial : public Benchmark {
protected:
    int n;
    std::vector<T> data[2];
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        n = config.get_int("n");
        assert_info((n & (n - 1)) == 0, "n should be a power of 2");
        workload = n * n * n;
        data[0].resize(n * n * n);
        data[1].resize(n * n * n);
    }

    const T &get_entry(int l, int i, int j, int k) const {
        return data[l][i * n * n + j * n + k];
    }

    T &get_entry(int l, int i, int j, int k) {
        return data[l][i * n * n + j * n + k];
    }

    bool test() const override {
        Config cfg;
        cfg.set("n", 128);
        JacobiBruteForce<T> bf;
        JacobiSerial<T> self;
        bf.initialize(cfg);
        bf.setup();
        bf.iterate();
        self.initialize(cfg);
        self.setup();
        self.iterate();
        bool same = true;
        for (int i = 0; i < self.n; i++) {
            for (int j = 0; j < self.n; j++) {
                for (int k = 0; k < self.n; k++) {
                    T a = self.get_entry(0, i, j, k), b = bf.data[0][i][j][k];
                    if (std::abs(a - b) / std::max((T)1e-3, std::max(std::abs(a), std::abs(b))) > 1e-3f) {
                        same = false;
                    }
                }
            }
        }
        self.finalize();
        return same;
    }

    virtual void setup() override {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    get_entry(0, i, j, k) = get_initial_entry<T>(i, j, k);
    }

    virtual void iterate() override {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++) {
                    T t(0);
                    if (i > 0)
                        t += get_entry(0, i - 1, j, k);
                    if (j > 0)
                        t += get_entry(0, i, j - 1, k);
                    if (k > 0)
                        t += get_entry(0, i, j, k - 1);
                    if (i + 1 < n)
                        t += get_entry(0, i + 1, j, k);
                    if (j + 1 < n)
                        t += get_entry(0, i, j + 1, k);
                    if (k + 1 < n)
                        t += get_entry(0, i, j, k + 1);
                    get_entry(1, i, j, k) = t / T(6.0);
                }
        data[0].swap(data[1]);
    }
};

#define REGISTER(B, name) \
    typedef B<float> B ## 32; \
    typedef B<double> B ## 64; \
    TC_IMPLEMENTATION(Benchmark, B ## 32, (name "_32")); \
    TC_IMPLEMENTATION(Benchmark, B ## 64, (name "_64"));

REGISTER(JacobiBruteForce, "jacobi_bf")
REGISTER(JacobiSerial, "jacobi_serial")


TC_NAMESPACE_END
