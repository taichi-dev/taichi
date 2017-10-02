/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <xmmintrin.h>
#include <immintrin.h>
#include <taichi/system/benchmark.h>
#include <taichi/system/threading.h>

#ifndef TC_DISABLE_SSE
#define TC_USE_SSE
#endif

#ifdef TC_USE_SSE
// #define TC_USE_AVX
#define REGISTER_32(B, name) \
  typedef B<float> B##32;    \
  TC_IMPLEMENTATION(Benchmark, B##32, (name "_32"));

#define REGISTER_64(B, name) \
  typedef B<double> B##64;   \
  TC_IMPLEMENTATION(Benchmark, B##64, (name "_64"));

#define REGISTER(B, name) \
  REGISTER_32(B, name)    \
  REGISTER_64(B, name)

TC_NAMESPACE_BEGIN

template <typename T>
T get_initial_entry(int i, int j, int k) {
  return (unsigned(i * 10000000007 + j * 321343212 + k * 412344739)) * T(1e-10);
}

template <typename T>
class JacobiSerial;

template <typename T>
class JacobiSIMD;

template <typename T>
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
  friend JacobiSIMD<T>;
};

REGISTER(JacobiBruteForce, "jacobi_bf")

template <typename T>
class JacobiSerial : public Benchmark {
 protected:
  int n;
  int ignore;

  void (JacobiSerial<T>::*iteration_method)();

  std::vector<T> data[2];
  Config cfg;

 public:
  void initialize(const Config &config) override {
    cfg = config;
    Benchmark::initialize(config);
    n = config.get_int("n");
    assert_info((n & (n - 1)) == 0, "n should be a power of 2");
    std::string method = config.get_string("iteration_method");
    ignore = config.get_int("ignore_boundary");
    workload = (n - ignore * 2) * (n - ignore * 2) * (n - ignore * 2);
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
    } else if (method == "relative_noif_inc_unroll2") {
      iteration_method = &JacobiSerial<T>::iterate_relative_noif_inc_unroll2;
    } else if (method == "relative_noif_inc_unroll4") {
      iteration_method = &JacobiSerial<T>::iterate_relative_noif_inc_unroll4;
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
    cfg.set("ignore_boundary", ignore);
    JacobiBruteForce<T> bf;
    JacobiSerial<T> self;
    bf.initialize(cfg);
    bf.setup();
    bf.iterate();
    self.initialize(cfg);
    self.setup();
    self.iterate();
    bool same = true;
    for (int i = ignore; i < self.n - ignore; i++) {
      for (int j = ignore; j < self.n - ignore; j++) {
        for (int k = ignore; k < self.n - ignore; k++) {
          T a = self.get_entry(0, i, j, k), b = bf.data[0][i][j][k];
          if (std::abs(a - b) /
                  std::max((T)1e-3, std::max(std::abs(a), std::abs(b))) >
              1e-3_f) {
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
    if (boundary > ignore)
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
    if (boundary > ignore)
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

  void iterate_relative_noif_inc_unroll2() {
    const int boundary = 4;
    if (boundary > ignore)
      iterate_boundary(boundary);
    const int b1 = boundary, b2 = n - boundary;
    for (int i = b1; i < b2; i++) {
      for (int j = b1; j < b2; j++) {
        int p = i * n * n + j * n + b1;
        T *p_i_minus = &data[0][0] + p - n * n;
        T *p_i_plus = &data[0][0] + p + n * n;
        T *p_j_minus = &data[0][0] + p - n;
        T *p_j_plus = &data[0][0] + p + n;
        T *p_k_minus = &data[0][0] + p - 1;
        T *p_k_plus = &data[0][0] + p + 1;
        for (int k = b1; k < b2; k += 2) {
          T a, b, c;
#define UNROLL                             \
  a = *p_i_minus + *p_i_plus;              \
  b = *p_j_minus + *p_j_plus;              \
  c = *p_k_minus + *p_k_plus;              \
  data[1][p] = ((a + b) + c) * T(1 / 6.0); \
  p++;                                     \
  p_i_minus++;                             \
  p_j_minus++;                             \
  p_k_minus++;                             \
  p_i_plus++;                              \
  p_j_plus++;                              \
  p_k_plus++;
          UNROLL
          UNROLL
        }
      }
    }
  }

  void iterate_relative_noif_inc_unroll4() {
    const int boundary = 4;
    if (boundary > ignore)
      iterate_boundary(boundary);
    const int b1 = boundary, b2 = n - boundary;
    for (int i = b1; i < b2; i++) {
      for (int j = b1; j < b2; j++) {
        int p = i * n * n + j * n + b1;
        T *p_i_minus = &data[0][0] + p - n * n;
        T *p_i_plus = &data[0][0] + p + n * n;
        T *p_j_minus = &data[0][0] + p - n;
        T *p_j_plus = &data[0][0] + p + n;
        T *p_k_minus = &data[0][0] + p - 1;
        T *p_k_plus = &data[0][0] + p + 1;
        for (int k = b1; k < b2; k += 4) {
          T a, b, c;
#define UNROLL                             \
  a = *p_i_minus + *p_i_plus;              \
  b = *p_j_minus + *p_j_plus;              \
  c = *p_k_minus + *p_k_plus;              \
  data[1][p] = ((a + b) + c) * T(1 / 6.0); \
  p++;                                     \
  p_i_minus++;                             \
  p_j_minus++;                             \
  p_k_minus++;                             \
  p_i_plus++;                              \
  p_j_plus++;                              \
  p_k_plus++;
          UNROLL
          UNROLL
          UNROLL
          UNROLL
        }
      }
    }
  }
};

REGISTER(JacobiSerial, "jacobi_serial")

template <typename T>
class JacobiSIMD : public Benchmark {
 protected:
  int n;
  int ignore;
  int num_threads;

  void (JacobiSIMD<T>::*iteration_method)();

  std::vector<T> data_[2];
  T *data[2];
  Config cfg;

 public:
  void initialize(const Config &config) override {
    cfg = config;
    Benchmark::initialize(config);
    n = config.get_int("n");
    ignore = config.get_int("ignore_boundary");
    std::string method = config.get_string("iteration_method");
    num_threads = config.get("num_threads", 1);
    assert_info((n & (n - 1)) == 0, "n should be a power of 2");
    workload = (n - ignore * 2) * (n - ignore * 2) * (n - ignore * 2);
    for (int i = 0; i < 2; i++)
      data_[i].resize(n * n * n + 32);
    for (int i = 0; i < 2; i++) {
      const int alignment = 64;
      size_t p = (size_t)&data_[i][0];
      assert(p % sizeof(T) == 0);
      data[i] =
          reinterpret_cast<T *>(p + (alignment - sizeof(T) -
                                     (p + alignment - sizeof(T)) % alignment));
      assert(reinterpret_cast<size_t>(data[i]) % alignment == 0);
    }
    if (method == "sse") {
      iteration_method = &JacobiSIMD<T>::iterate_sse;
    } else if (method == "sse_prefetch") {
      iteration_method = &JacobiSIMD<T>::iterate_sse_prefetch;
    } else if (method == "sse_block") {
      iteration_method = &JacobiSIMD<T>::iterate_sse_block;
    } else if (method == "sse_threaded") {
      iteration_method = &JacobiSIMD<T>::iterate_sse_threaded;
    } else if (method == "avx") {
      iteration_method = &JacobiSIMD<T>::iterate_avx;
    } else {
      error("Iteration method not found: " + method);
    }
  }

  ~JacobiSIMD() {}

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
    cfg.set("ignore_boundary", ignore);
    cfg.set("num_threads", num_threads);
    JacobiBruteForce<T> bf;
    JacobiSIMD<T> self;
    bf.initialize(cfg);
    bf.setup();
    bf.iterate();
    self.initialize(cfg);
    self.setup();
    self.iterate();
    bool same = true;
    for (int i = ignore; i < self.n - ignore; i++) {
      for (int j = ignore; j < self.n - ignore; j++) {
        for (int k = ignore; k < self.n - ignore; k++) {
          T a = self.get_entry(0, i, j, k), b = bf.data[0][i][j][k];
          if (std::abs(a - b) /
                  std::max((T)1e-3, std::max(std::abs(a), std::abs(b))) >
              1e-3_f) {
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
    std::swap(data[0], data[1]);
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

  void iterate_sse();

  void iterate_sse_threaded();

  void iterate_sse_prefetch();

  void iterate_sse_block();

  void iterate_avx();
};

#define FUSION_32_SSE                     \
  minus_k = _mm_loadu_ps(src + p - 1);    \
  plus_k = _mm_loadu_ps(src + p + 1);     \
  minus_i = _mm_load_ps(src + p - n * n); \
  plus_i = _mm_load_ps(src + p + n * n);  \
  minus_j = _mm_load_ps(src + p - n);     \
  plus_j = _mm_load_ps(src + p + n);      \
  plus_k = _mm_add_ps(minus_k, plus_k);   \
  plus_j = _mm_add_ps(minus_j, plus_j);   \
  plus_k = _mm_add_ps(plus_j, plus_k);    \
  plus_i = _mm_add_ps(minus_i, plus_i);   \
  plus_k = _mm_add_ps(plus_i, plus_k);    \
  plus_k = _mm_mul_ps(plus_k, c);         \
  _mm_store_ps(dst + p, plus_k);

template <>
void JacobiSIMD<float>::iterate_sse() {
  const int boundary = 4;
  const float one_over_six = float(1) / 6;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const int b1 = boundary, b2 = n - boundary;
  float *__restrict src = data[0];
  float *__restrict dst = data[1];
  __m128 plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
  __m128 c = _mm_broadcast_ss(&one_over_six);
  for (int i = b1; i < b2; i++) {
    for (int j = b1; j < b2; j++) {
      int p_base = i * n * n + j * n;
      for (int k = b1; k < b2; k += 4) {
        int p = p_base + k;
        FUSION_32_SSE
      }
    }
  }
}

template <>
void JacobiSIMD<double>::iterate_sse() {
  error("not implemented");
}

template <>
void JacobiSIMD<float>::iterate_sse_prefetch() {
  const int boundary = 4;
  const float one_over_six = float(1) / 6;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const int b1 = boundary, b2 = n - boundary;
  float *__restrict src = data[0];
  float *__restrict dst = data[1];
  __m128 plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
  __m128 c = _mm_broadcast_ss(&one_over_six);
  for (int i = b1; i < b2; i++) {
    for (int j = b1; j < b2; j++) {
      int p_base = i * n * n + j * n;
      for (int k = b1; k < b2; k += 4) {
        _mm_prefetch((const char *)src + p_base - n * n + n + k, _MM_HINT_T0);
        _mm_prefetch((const char *)src + p_base + n * n + n + k, _MM_HINT_T0);
        _mm_prefetch((const char *)src + p_base + n * 2 + k, _MM_HINT_T0);
        _mm_prefetch((const char *)src + p_base + k, _MM_HINT_T0);
        int p = p_base + k;
        FUSION_32_SSE
      }
    }
  }
}

template <>
void JacobiSIMD<double>::iterate_sse_prefetch() {
  error("not implemented");
}

template <>
void JacobiSIMD<float>::iterate_sse_block() {
  const int boundary = 8;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const float one_over_six = float(1) / 6;
  const int b1 = boundary, b2 = n - boundary;
  float *__restrict src = data[0];
  float *__restrict dst = data[1];
  __m128 plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
  __m128 c = _mm_broadcast_ss(&one_over_six);
  const int bi = 16;
  const int bj = 16;
  const int bk = 16;
  for (int i = b1; i < b2; i++) {
    for (int j = b1; j < b2; j += bj) {
      for (int k = b1; k < b2; k += bk) {
        for (int ii = 0; ii < bi; ii++) {
          for (int jj = 0; jj < bj; jj++) {
            for (int kk = 0; kk < bk; kk += 4) {
              int p = (i + ii) * n * n + (j + jj) * n + k + kk;
              FUSION_32_SSE
            }
          }
        }
      }
    }
  }
}

template <>
void JacobiSIMD<float>::iterate_sse_threaded() {
  const int boundary = 4;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const int b1 = boundary, b2 = n - boundary;
  float *__restrict src = data[0];
  float *__restrict dst = data[1];
  ThreadedTaskManager::run(
      b1, b2, num_threads, [src, dst, b1, b2, this](int i) {
        const float one_over_six = float(1) / 6;
        __m128 plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
        __m128 c = _mm_broadcast_ss(&one_over_six);
        for (int j = b1; j < b2; j++) {
          int p_base = i * n * n + j * n;
          for (int k = b1; k < b2; k += 4) {
            int p = p_base + k;
            FUSION_32_SSE
          }
        }
      });
}

template <>
void JacobiSIMD<double>::iterate_sse_threaded() {
  error("not implemented");
}

template <>
void JacobiSIMD<double>::iterate_sse_block() {
  error("not implemented");
}

template <>
void JacobiSIMD<float>::iterate_avx() {
#ifdef TC_USE_AVX
  const int boundary = 8;
  const float one_over_six = float(1) / 6;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const int b1 = boundary, b2 = n - boundary;
  float *__restrict src = data[0];
  float *__restrict dst = data[1];
  __m256 plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
  __m256 c = _mm256_broadcast_ss(&one_over_six);
  for (int i = b1; i < b2; i++) {
    for (int j = b1; j < b2; j++) {
      for (int k = b1; k < b2; k += 8) {
        int p = i * n * n + j * n + k;
        minus_k = _mm256_loadu_ps(src + p - 1);
        plus_k = _mm256_loadu_ps(src + p + 1);
        minus_i = _mm256_load_ps(src + p - n * n);
        plus_i = _mm256_load_ps(src + p + n * n);
        minus_j = _mm256_load_ps(src + p - n);
        plus_j = _mm256_load_ps(src + p + n);
        plus_k = _mm256_add_ps(minus_k, plus_k);
        plus_j = _mm256_add_ps(minus_j, plus_j);
        plus_k = _mm256_add_ps(plus_j, plus_k);
        plus_i = _mm256_add_ps(minus_i, plus_i);
        plus_k = _mm256_add_ps(plus_i, plus_k);
        plus_k = _mm256_mul_ps(plus_k, c);
        _mm256_store_ps(dst + p, plus_k);
      }
    }
  }
#else
  error("not implemented (needs AVX support)");
#endif
}

template <>
void JacobiSIMD<double>::iterate_avx() {
  const int boundary = 4;
  const double one_over_six = double(1) / 6;
  if (boundary > ignore)
    iterate_boundary(boundary);
  const int b1 = boundary, b2 = n - boundary;
  double *__restrict src = data[0];
  double *__restrict dst = data[1];
  __m256d plus_i, minus_i, plus_j, minus_j, plus_k, minus_k;
  __m256d c = _mm256_broadcast_sd(&one_over_six);
  for (int i = b1; i < b2; i++) {
    for (int j = b1; j < b2; j++) {
      int p_base = i * n * n + j * n;
      for (int k = b1; k < b2; k += 4) {
        int p = p_base + k;
        minus_k = _mm256_loadu_pd(src + p - 1);
        plus_k = _mm256_loadu_pd(src + p + 1);
        minus_i = _mm256_load_pd(src + p - n * n);
        plus_i = _mm256_load_pd(src + p + n * n);
        minus_j = _mm256_load_pd(src + p - n);
        plus_j = _mm256_load_pd(src + p + n);
        plus_k = _mm256_add_pd(minus_k, plus_k);
        plus_j = _mm256_add_pd(minus_j, plus_j);
        plus_k = _mm256_add_pd(plus_j, plus_k);
        plus_i = _mm256_add_pd(minus_i, plus_i);
        plus_k = _mm256_add_pd(plus_i, plus_k);
        plus_k = _mm256_mul_pd(plus_k, c);
        _mm256_store_pd(dst + p, plus_k);
      }
    }
  }
}

REGISTER(JacobiSIMD, "jacobi_simd")

TC_NAMESPACE_END

#endif
