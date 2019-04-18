#pragma once

#define FUNC_DECL

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace Tlang {
#define TLANG_NAMESPACE_END \
  }                         \
  }

#include "context.h"

#include <atomic>
#include <numeric>
#include <mutex>
#include <unordered_map>
#include <iostream>

#if !defined(TC_INCLUDED)

#ifdef _WIN64
#define TC_FORCE_INLINE __forceinline
#else
#define TC_FORCE_INLINE inline __attribute__((always_inline))
#endif
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <array>
#include <vector>
#include <omp.h>

using float32 = float;
using float64 = double;
using int32 = int;
using uint32 = unsigned int;
using uint64 = unsigned long long;
using uint8 = unsigned char;
using uint16 = unsigned short;

#if defined(TLANG_GPU)
#include <cuda_runtime.h>
#undef FUNC_DECL
#define FUNC_DECL __host__ __device__
#endif

#define TC_ASSERT(x)                                                    \
  if (!(x)) {                                                           \
    std::cout << "Ln" << __LINE__ << ":" << #x << ": Assertion failed." \
              << std::endl;                                             \
    exit(-1);                                                           \
  }
#define TC_P(x)                                                          \
  std::cout << __FILE__ << "@" << __LINE__ << ": " << #x << " = " << (x) \
            << std::endl;
namespace taichi {
TC_FORCE_INLINE uint32 rand_int() noexcept {
  static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
  unsigned int t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

TC_FORCE_INLINE uint64 rand_int64() noexcept {
  return ((uint64)rand_int() << 32) + rand_int();
}

template <typename T>
TC_FORCE_INLINE T rand() noexcept;

template <>
TC_FORCE_INLINE float rand<float>() noexcept {
  return rand_int() * (1.0f / 4294967296.0f);
}

template <>
TC_FORCE_INLINE double rand<double>() noexcept {
  return rand_int() * (1.0 / 4294967296.0);
}

template <>
TC_FORCE_INLINE int rand<int>() noexcept {
  return rand_int();
}

template <typename T>
TC_FORCE_INLINE T rand() noexcept;
}  // namespace taichi

#endif

#if !defined(TC_GPU)
#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif
#endif

TLANG_NAMESPACE_BEGIN

constexpr int max_num_indices = 4;
constexpr int max_num_snodes = 1024;

template <typename T, typename G>
__device__ T union_cast(G g) {
  static_assert(sizeof(T) == sizeof(G), "");
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

template <typename T, typename G>
T union_cast_different_size(G g) {
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

TLANG_NAMESPACE_END

#if defined(TC_GPU)
__device__ __constant__ void **device_head;
__device__ __constant__ void *device_data;
#endif
