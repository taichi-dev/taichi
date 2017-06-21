/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math_util.h>

// >= AVX 2
#include <immintrin.h>

#ifdef _WIN64
#define TC_ALIGNED(x) __declspec(align(x))
#else
#define TC_ALIGNED(x) __attribute__((aligned(x)))
#endif


TC_NAMESPACE_BEGIN

// SIMD Vector4
struct TC_ALIGNED(16) Vector4s {
    union {
        __m128 v;
        struct {
            float x, y, z, w;
        };
    };

    Vector4s() : Vector4s(0.0f) {};

    Vector4s(const Vector4 &vec) : Vector4s(vec.x, vec.y, vec.z, vec.w) {}

    Vector4s(const Vector3 &vec, float w = 0.0f) : Vector4s(vec.x, vec.y, vec.z, w) {}

    Vector4s(real x, real y, real z, real w) : v(_mm_set_ps(w, z, y, x)) {}

    Vector4s(real x) : v(_mm_set1_ps(x)) {}

    Vector4s(__m128 v) : v(v) {}

    float &operator[](int i) { return (&x)[i]; }

    const float &operator[](int i) const { return (&x)[i]; }

    operator __m128() const { return v; }

    operator __m128i() const { return _mm_castps_si128(v); }

    operator __m128d() const { return _mm_castps_pd(v); }

    Vector4s &operator=(const Vector4s o) {
        v = o.v;
        return *this;
    }

    Vector4s operator+(const Vector4s &o) { return _mm_add_ps(v, o.v); }

    Vector4s operator-(const Vector4s &o) { return _mm_sub_ps(v, o.v); }

    Vector4s operator*(const Vector4s &o) { return _mm_mul_ps(v, o.v); }

    Vector4s operator/(const Vector4s &o) { return _mm_div_ps(v, o.v); }

    Vector4s &operator+=(const Vector4s &o) {
        (*this) = (*this) + o;
        return *this;
    }

    Vector4s &operator-=(const Vector4s &o) {
        (*this) = (*this) - o;
        return *this;
    }

    Vector4s &operator*=(const Vector4s &o) {
        (*this) = (*this) * o;
        return *this;
    }

    Vector4s &operator/=(const Vector4s &o) {
        (*this) = (*this) / o;
        return *this;
    }

    Vector3 to_vec3() const {
        return Vector3(x, y, z);
    }


};

inline void print(const Vector4s v) {
    for (int i = 0; i < 4; i++) {
        printf("%9.4f ", v[i]);
    }
    printf("\n");
}


// Note: assuming abs(x) <= 2!!
inline real w(real x) {
    x = abs(x);
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
        Vector4s ret;
        Vector4s w_cache[3];
        Vector4s dw_cache[3];


        /*
        inline real w(real x) {
            x = abs(x);
            if (x < 1) {
                return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
            } else {
                return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
            }
        }

        inline real dw(real x) {
            real s = x < 0.0f ? -1.0f : 1.0f;
            x *= s;
            real val;
            real xx = x * x;
            if (x < 1.0f) {
                val = 1.5f * xx - 2.0f * x;
            } else {
                val = -0.5f * xx + 2.0f * x - 2.0f;
            }
            return s * val;
        }
        */

        // [(2 - x), (1 - x), x, x + 1]
        for (int k = 0; k < 3; k++) {
            auto t = Vector4s(p[k]) - Vector4s(3, 2, 1, 0);
            auto tt = t * t;
            auto ttt = tt * t;
            w_cache[k] = Vector4s(1 / 6.0f, -0.5f, 0.5f, -1 / 6.0f) * ttt +
                         Vector4s(1, -1, -1, 1) * tt +
                         Vector4s(2, 0, 0, -2) * t +
                         Vector4s(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f);
            dw_cache[k] = Vector4s(0.5f, -1.5f, 1.5f, -0.5f) * tt +
                          Vector4s(2, -2, -2, 2) * t +
                          Vector4s(2, 0, 0, -2);
        }
        Vector4s w_stages[3][4];
        for (int k = 0; k < 4; k++) {
            w_stages[0][k] = Vector4s(dw_cache[0][k], w_cache[0][k], w_cache[0][k], w_cache[0][k]);
            w_stages[1][k] = Vector4s(w_cache[1][k], dw_cache[1][k], w_cache[1][k], w_cache[1][k]);
            w_stages[2][k] = Vector4s(w_cache[2][k], w_cache[2][k], dw_cache[2][k], w_cache[2][k]);
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                auto p = w_stages[0][i] * w_stages[1][j];
                for (int k = 0; k < 4; k++) {
                    auto gw = p * w_stages[2][k];
                    real weight = gw.w;
                    ret += Vector4s(weight) * gw;
                }
            }
        }
        return ret.x * 2 + ret.y * 3 + ret.z * 4 + ret.w * 5;
    }

    real sum_brute_force(Vector3 p) const {
        Vector4 ret(0.0f);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    Vector3 d_pos = p - Vector3(i, j, k);
                    real weight = w(d_pos);
                    Vector3 gw = dw(d_pos);
                    ret += weight * Vector4(gw, weight);
                }
            }
        }
        return ret.x * 2 + ret.y * 3 + ret.z * 4 + ret.w * 5;
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
            if (abs(bf_result - simd_result) > 1e-6) {
                printf("%f %f\n", bf_result, simd_result);
                error("value mismatch");
            }
        }
        return true;
    }
};

TC_IMPLEMENTATION(Benchmark, KernelCalculationBenchmark, "mpm_kernel");

TC_NAMESPACE_END
