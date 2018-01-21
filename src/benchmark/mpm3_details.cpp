/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

// Note: assuming abs(x) <= 2!!
inline real w(real x) {
  x = abs(x);
  if (x < 1) {
    return 0.5_f * x * x * x - x * x + 2.0_f / 3.0_f;
  } else {
    return -1.0_f / 6.0_f * x * x * x + x * x - 2 * x + 4.0_f / 3.0_f;
  }
}

// Note: assuming abs(x) <= 2!!
inline real dw(real x) {
  real s = x < 0.0_f ? -1.0_f : 1.0_f;
  x *= s;
  real val;
  real xx = x * x;
  if (x < 1.0_f) {
    val = 1.5_f * xx - 2.0_f * x;
  } else {
    val = -0.5f * xx + 2.0_f * x - 2.0_f;
  }
  return s * val;
}

inline real w(const Vector3 &a) {
  return w(a.x) * w(a.y) * w(a.z);
}

inline Vector3 dw(const Vector3 &a) {
  return Vector3(dw(a.x) * w(a.y) * w(a.z), w(a.x) * dw(a.y) * w(a.z),
                 w(a.x) * w(a.y) * dw(a.z));
}

class KernelCalculationBenchmark : public Benchmark {
 private:
  bool brute_force;
  std::vector<Vector3> input;

 public:
  void initialize(const Config &config) override {
    Benchmark::initialize(config);
    brute_force = config.get<bool>("brute_force");
    input.resize(workload);
    for (int i = 0; i < workload; i++) {
      input[i] = Vector3(rand(), rand(), rand()) + Vector3(1.0_f);
    }
  }

 protected:
  real sum_simd(Vector3 p) const {
    Vector4 ret;
    Vector4 w_cache[3];
    Vector4 dw_cache[3];

    // [x, x - 1, x - 2, x - 3]
    // [+,     +,     -,     -]
    for (int k = 0; k < 3; k++) {
      const Vector4 t = Vector4(p[k]) - Vector4(0, 1, 2, 3);
      auto tt = t * t;
      auto ttt = tt * t;
      w_cache[k] = Vector4(-1 / 6.0f, 0.5f, -0.5f, 1 / 6.0f) * ttt +
                   Vector4(1, -1, -1, 1) * tt + Vector4(-2, 0, 0, 2) * t +
                   Vector4(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f);
      dw_cache[k] = Vector4(-0.5f, 1.5_f, -1.5_f, 0.5f) * tt +
                    Vector4(2, -2, -2, 2) * t + Vector4(-2, 0, 0, 2);
      /*
      for (int i = 0; i < 4; i++) {
          TC_P(dw_cache[k][i]);
          TC_P(dw(p[k] - i));
      }
      */
      /* FMA - doesn't help...
      Vector4 w(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f), dw(2, 0, 0, -2);
      const Vector4 tt = t * t;
      const Vector4 ttt = tt * t;
      w = fused_mul_add(Vector4(2, 0, 0, -2), t, w);
      w = fused_mul_add(Vector4(1, -1, -1, 1), tt, w);
      w = fused_mul_add(Vector4(1 / 6.0f, -0.5f, 0.5f, -1 / 6.0f), ttt, w);
      dw = fused_mul_add(Vector4(2, -2, -2, 2), t, dw);
      dw = fused_mul_add(Vector4(0.5f, -1.5_f, 1.5_f, -0.5f), tt, dw);
      w_cache[k] = w;
      dw_cache[k] = dw;
       */
    }

    Vector4 w_stages[3][4];
    for (int k = 0; k < 4; k++) {
      w_stages[0][k] =
          Vector4(dw_cache[0][k], w_cache[0][k], w_cache[0][k], w_cache[0][k]);
      w_stages[1][k] =
          Vector4(w_cache[1][k], dw_cache[1][k], w_cache[1][k], w_cache[1][k]);
      w_stages[2][k] =
          Vector4(w_cache[2][k], w_cache[2][k], dw_cache[2][k], w_cache[2][k]);
    }

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        auto p = w_stages[0][i] * w_stages[1][j];
        for (int k = 0; k < 4; k++) {
          auto gw = p * w_stages[2][k];
          real weight = gw.w;
          ret += Vector4(weight) * gw;
        }
      }
    }
    return ret.x * 2 + ret.y * 3 + ret.z * 4 + ret.w * 5;
  }

  real sum_brute_force(Vector3 p) const {
    Vector4 ret(0.0_f);
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
    real ret = 0.0_f;
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
        TC_ERROR("value mismatch");
      }
    }
    return true;
  }
};

TC_IMPLEMENTATION(Benchmark, KernelCalculationBenchmark, "mpm_kernel");

TC_NAMESPACE_END
