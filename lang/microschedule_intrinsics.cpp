#if (0)
#include <taichi/util.h>
#include "util.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

void mul0(int n, float *a, float *b, float *c, float *d) {
  for (int i = 0; i < n;) {
//#define LOOP                                                             \
  _mm256_store_ps(d + i, _mm256_set1_ps(2.0_f) + _mm256_load_ps(b + i)); \
  i += 8;
#define LOOP                                     \
  _mm256_store_ps(d + i, _mm256_set1_ps(2.0_f)); \
  i += 8;
    LOOP LOOP LOOP LOOP LOOP LOOP LOOP LOOP
  }
}

void mul1(int n, float *a, float *b, float *c, float *d) {
  /*
  for (int i = 0; i < n; i++) {
    auto va = _mm256_broadcast_ss(a + i);
    auto vb = _mm256_broadcast_ss(b + i);
    _mm256_store_ps(d + 8 * i, va * vb * _mm256_load_ps(c + 8 * i));
  }
  */
  for (int i = 0; i < n; i += 4) {
#define LOOP(l)                                                                \
  {                                                                            \
    auto va = _mm256_broadcast_ss(a + i + l);                                  \
    auto vb = _mm256_broadcast_ss(b + i + l);                                  \
    _mm256_store_ps(d + 8 * (i + l),                                           \
                    va * va * va * va * va * _mm256_load_ps(c + 8 * (i + l))); \
  }
    LOOP(0);
    LOOP(1);
    LOOP(2);
    LOOP(3);
  }
}

void mul2(int n, float *a, float *b, float *c, float *d) {
  for (int i = 0; i < n; i += 8) {
    auto va = _mm256_load_ps(a + i);
    auto vb = _mm256_load_ps(b + i);
    auto vab = va / vb / vb / vb;
#define LOOP(l)                    \
  _mm256_store_ps(d + 8 * (i + l), \
                  _mm256_set1_ps(vab[l]) * _mm256_load_ps(c + 8 * (i + l)));
    LOOP(0);
    LOOP(1);
    LOOP(2);
    LOOP(3);
    LOOP(4);
    LOOP(5);
    LOOP(6);
    LOOP(7);
  }
}

void mul3(int n, float *a, float *b, float *c, float *d) {
  for (int i = 0; i < n;) {
#define LOOP(l)                                                              \
  _mm256_store_ps(d + 8 * (i + l),                                           \
                  _mm256_set1_ps(vab[l]) * _mm256_load_ps(c + 8 * (i + l))); \
  _mm256_store_ps(                                                           \
      d + 8 * (i + 8 + l),                                                   \
      _mm256_set1_ps(vab2[l]) * _mm256_load_ps(c + 8 * (i + l + 8)));
    {
      float buffer[16];
      auto va = _mm256_load_ps(a + i);
      auto va2 = _mm256_load_ps(a + i + 8);
      // auto vb = _mm256_load_ps(b + i);
      auto vab = va * va * va * va * va;
      auto vab2 = va2 * va2 * va2 * va2 * va2;
      LOOP(0);
      LOOP(1);
      LOOP(2);
      LOOP(3);
      LOOP(4);
      LOOP(5);
      LOOP(6);
      LOOP(7);
      i += 16;
    }
  }
}

auto benchmark_microschedule = []() {
  int n = 512;
  AlignedAllocator A(sizeof(float32) * n);
  AlignedAllocator B(sizeof(float32) * n);
  AlignedAllocator C(sizeof(float32) * n * 8);
  AlignedAllocator D(sizeof(float32) * n * 8);

  for (int i = 0; i < n; i++) {
    A.get<float32>()[i] = 1;      //(i + 1);
    B.get<float32>()[i] = 2.0_f;  // / (i + 1);
    for (int j = 0; j < 8; j++) {
      C.get<float32>()[i * 8 + j] = j;
    }
  }

  auto scal = measure_cpe(
      [&]() {
        mul0(n, A.get<float32>(), B.get<float32>(), C.get<float32>(),
             D.get<float32>());
      },
      n, 1);

  TC_P(scal);

  auto pure = measure_cpe(
      [&]() {
        mul1(n, A.get<float32>(), B.get<float32>(), C.get<float32>(),
             D.get<float32>());
      },
      n, 1);

  TC_P(pure);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 8; j++) {
      real val = D.get<float32>()[i * 8 + j];
      real gt = 2 * j;
      // TC_WARN_UNLESS(std::abs(val - gt) < 1e-6f, "");
      D.get<float32>()[i * 8 + j] = 0;  // for second run
    }
  }

  auto micro = measure_cpe(
      [&]() {
        mul3(n, A.get<float32>(), B.get<float32>(), C.get<float32>(),
             D.get<float32>());
      },
      n, 1);
  TC_P(micro);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 8; j++) {
      real val = D.get<float32>()[i * 8 + j];
      real gt = 2 * j;
      // TC_WARN_UNLESS(std::abs(val - gt) < 1e-6f, "");
    }
  }

};

TC_REGISTER_TASK(benchmark_microschedule);
}  // namespace Tlang
TC_NAMESPACE_END
#endif
