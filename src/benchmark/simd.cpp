/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

class Matrix4sBenchmark : public Benchmark {
 private:
  int n;
  bool brute_force;
  std::vector<Vector4> input;
  std::vector<Vector4> input_s;
  Matrix4 M;

 public:
  void initialize(const Config &config) override {
    Benchmark::initialize(config);
    brute_force = config.get<bool>("brute_force");
    input.resize(workload);
    input_s.resize(workload);
    for (int i = 0; i < workload; i++) {
      input[i] = Vector4(rand(), rand(), rand(), rand());
      input_s[i] = input[i];
    }
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        M[i][j] = rand();
      }
    }
  }

 protected:
  void iterate() override {
    if (brute_force) {
      Vector4 ret(0.0_f);
      Matrix4 Ms(M);
      for (int i = 0; i < workload; i++) {
        ret += Ms * input_s[i];
      }
      dummy = (int)(ret.length());
    } else {
      Vector4 ret(0.0_f);
      for (int i = 0; i < workload; i++) {
        ret += M * input[i];
      }
      dummy = (int)(length(ret));
    }
  }

 public:
  bool test() const override {
    Matrix4 Ms(M);
    for (int i = 0; i < workload; i++) {
      Vector4 bf_result = M * input[i];
      Vector4 simd_result = Ms * input_s[i];
      if ((bf_result - simd_result).length() > 1e-6) {
        P(M);
        P(Ms);
        P(input[i]);
        P(input_s[i]);
        P(i);
        P(bf_result);
        P(simd_result);
        error("value mismatch");
      }
    }
    return true;
  }
};

TC_IMPLEMENTATION(Benchmark, Matrix4sBenchmark, "matrix4s");

TC_NAMESPACE_END
