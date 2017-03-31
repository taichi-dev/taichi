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
    return (unsigned(i * 10000000007 + j * 321343212 + k * 412344739)) * T(1e-10);
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

    void (JacobiSerial<T>::*iteration_method)();

    std::vector<T> data[2];
    Config cfg;
public:
    void initialize(const Config &config) override {
        cfg = config;
        Benchmark::initialize(config);
        n = config.get_int("n");
        std::string method = config.get_string("iteration_method");
        assert_info((n & (n - 1)) == 0, "n should be a power of 2");
        workload = n * n * n;
        data[0].resize(n * n * n);
        data[1].resize(n * n * n);
        if (method == "naive") {
            iteration_method = &JacobiSerial<T>::iterate_naive;
        } else if (method == "relative") {
            iteration_method = &JacobiSerial<T>::iterate_relative;
        } else if (method == "relative_noif") {
            iteration_method = &JacobiSerial<T>::iterate_relative_noif;
        } else if (method == "relative_noif_inc") {
            iteration_method = &JacobiSerial<T>::iterate_relative_noif_inc;
        } else if (method == "relative_noif_inc_unroll") {
            iteration_method = &JacobiSerial<T>::iterate_relative_noif_inc_unroll;
        } else {
            error("Iteration method not found: " + method);
        }
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
        cfg.set("iteration_method", this->cfg.get_string("iteration_method"));
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
        ((*this).*(this->iteration_method))();
        data[0].swap(data[1]);
    }

    void iterate_naive() {
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
                    get_entry(1, i, j, k) = t * T(1 / 6.0);
                }
    }

    void iterate_relative() {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++) {
                    int p = i * n * n + j * n + k;
                    T t(0);
                    if (i > 0)
                        t += data[0][p - n * n];
                    if (j > 0)
                        t += data[0][p - n];
                    if (k > 0)
                        t += data[0][p - 1];
                    if (i + 1 < n)
                        t += data[0][p + n * n];
                    if (j + 1 < n)
                        t += data[0][p + n];
                    if (k + 1 < n)
                        t += data[0][p + 1];
                    get_entry(1, i, j, k) = t * T(1 / 6.0);
                }
    }

    void iterate_boundary(const int boundary) {
        const int b1 = boundary, b2 = n - boundary;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    int p = i * n * n + j * n + k;
                    T t(0);
                    if (i > 0)
                        t += data[0][p - n * n];
                    if (j > 0)
                        t += data[0][p - n];
                    if (k > 0)
                        t += data[0][p - 1];
                    if (i + 1 < n)
                        t += data[0][p + n * n];
                    if (j + 1 < n)
                        t += data[0][p + n];
                    if (k + 1 < n)
                        t += data[0][p + 1];
                    data[1][p] = t * T(1 / 6.0);
                    if (b1 <= std::min(i, j) && std::max(i, j) < b2) {
                        if (k == b1 - 1) {
                            k = b2 - 1;
                        }
                    }
                }
            }
        }
    }

    void iterate_relative_noif() {
        const int boundary = 4;
        iterate_boundary(boundary);
        const int b1 = boundary, b2 = n - boundary;
        for (int i = b1; i < b2; i++) {
            for (int j = b1; j < b2; j++) {
                for (int k = b1; k < b2; k++) {
                    int p = i * n * n + j * n + k;
                    T t(0);
                    t += data[0][p - n * n];
                    t += data[0][p - n];
                    t += data[0][p - 1];
                    t += data[0][p + n * n];
                    t += data[0][p + n];
                    t += data[0][p + 1];
                    data[1][p] = t * T(1 / 6.0);
                }
            }
        }
    }

    void iterate_relative_noif_inc() {
        const int boundary = 4;
        iterate_boundary(boundary);
        const int b1 = boundary, b2 = n - boundary;
        for (int i = b1; i < b2; i++) {
            for (int j = b1; j < b2; j++) {
                int p = i * n * n + j * n + b1;
                int p_i_minus = p - n * n;
                int p_j_minus = p - n;
                int p_i_plus = p + n * n;
                int p_j_plus = p + n;
                for (int k = b1; k < b2; k++) {
                    T t(0);
                    t += data[0][p - 1];
                    t += data[0][p + 1];
                    t += data[0][p_i_minus];
                    t += data[0][p_j_minus];
                    t += data[0][p_i_plus];
                    t += data[0][p_j_plus];
                    data[1][p] = t * T(1 / 6.0);
                    p++;
                    p_i_minus++;
                    p_j_minus++;
                    p_i_plus++;
                    p_j_plus++;
                }
            }
        }
    }

    void iterate_relative_noif_inc_unroll() {
        const int boundary = 4;
        iterate_boundary(boundary);
        const int b1 = boundary, b2 = n - boundary;
        for (int i = b1; i < b2; i++) {
            for (int j = b1; j < b2; j++) {
                int p = i * n * n + j * n + b1;
                T* p_i_minus = &data[0][0] + p - n * n;
                T* p_i_plus = &data[0][0] + p + n * n;
                T* p_j_minus = &data[0][0] + p - n;
                T* p_j_plus = &data[0][0] + p + n;
                T* p_k_minus = &data[0][0] + p - 1;
                T* p_k_plus = &data[0][0] + p + 1;
                for (int k = b1; k < b2; k += 2) {
                    T a, b, c;
#define UNROLL \
                    a = *p_i_minus + *p_i_plus; \
                    b = *p_j_minus + *p_j_plus; \
                    c = *p_k_minus + *p_k_plus; \
                    data[1][p] = ((a + b) + c) * T(1 / 6.0); \
                    p++; \
                    p_i_minus++; \
                    p_j_minus++; \
                    p_k_minus++; \
                    p_i_plus++; \
                    p_j_plus++; \
                    p_k_plus++;
                    UNROLL
                    UNROLL
                }
            }
        }
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
