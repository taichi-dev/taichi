/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math_util.h>

TC_NAMESPACE_BEGIN

// SIMD Vector4
class Vector4s {

};

// Note: assuming abs(x) <= 2!!
inline real w(real x) {
    x = abs(x);
#ifdef CV_ON
    assert(x <= 2);
#endif
    if (x < 1) {
        return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
    } else {
        return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
    }
}

// Note: assuming abs(x) <= 2!!
inline real dw(real x) {
    real s = x < 0.0f ? -1.0f : 1.0f;
    x *= s;
#ifdef CV_ON
    assert(x <= 2.0f);
#endif
    real val;
    real xx = x * x;
    if (x < 1.0f) {
        val = 1.5f * xx - 2.0f * x;
    } else {
        val = -0.5f * xx + 2.0f * x - 2.0f;
    }
    return s * val;
}

inline real w(const Vector3 &a) {
    return w(a.x) * w(a.y) * w(a.z);
}

inline Vector3 dw(const Vector3 &a) {
    return Vector3(dw(a.x) * w(a.y) * w(a.z), w(a.x) * dw(a.y) * w(a.z), w(a.x) * w(a.y) * dw(a.z));
}

class KernelCalculationBenchmark : public Benchmark {
private:
    int n;
    bool brute_force;
    std::vector<Vector3> input;
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        brute_force = config.get_bool("brute_force");
        input.resize(workload);
        for (int i = 0; i < workload; i++) {
            input[i] = Vector3(rand(), rand(), rand()) + Vector3(1.0f);
        }
    }

protected:
    real sum_simd(Vector3 p) const {
        real ret = 0.0f;
        real w_cache[3][4];
        real dw_cache[3][4];
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 4; i++) {
                w_cache[k][i] = w(p[k] - i);
                dw_cache[k][i] = dw(p[k] - i);
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    real weight = w_cache[0][i] * w_cache[1][j] * w_cache[2][k];
                    Vector3 gw = Vector3(
                            dw_cache[0][i] * w_cache[1][j] * w_cache[2][k],
                            w_cache[0][i] * dw_cache[1][j] * w_cache[2][k],
                            w_cache[0][i] * w_cache[1][j] * dw_cache[2][k]
                    );
                    ret += weight * (gw[0] + gw[1] + gw[2]);
                }
            }
        }
        return ret;
    }

    real sum_brute_force(Vector3 p) const {
        real ret = 0.0f;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    Vector3 d_pos = p - Vector3(i, j, k);
                    real weight = w(d_pos);
                    Vector3 gw = dw(d_pos);
                    ret += weight * (gw[0] + gw[1] + gw[2]);
                }
            }
        }
        return ret;
    }

    void iterate() override {
        real ret = 0.0f;
        if (brute_force) {
            for (int i = 0; i < workload; i++) {
                ret += sum_brute_force(input[i]);
            }
        } else {
            for (int i = 0; i < workload; i++) {
                ret += sum_simd(input[i]);
            }
        }
        dummy = (int)(ret);
    }

public:
    bool test() const override {
        for (int i = 0; i < workload; i++) {
            real bf_result = sum_brute_force(input[i]);
            real simd_result = sum_simd(input[i]);
            if (abs(bf_result - simd_result) > 1e-7) {
                printf("%f %f\n", bf_result, simd_result);
                error("value mismatch");
            }
        }
        return true;
    }
};

TC_IMPLEMENTATION(Benchmark, KernelCalculationBenchmark, "mpm_kernel");

TC_NAMESPACE_END
