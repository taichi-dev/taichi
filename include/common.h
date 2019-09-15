#pragma once

#define FUNC_DECL

#if defined(TLANG_GPU)
__device__ __constant__ void **device_head;
__device__ __constant__ void *device_data;
__device__ __constant__ int *error_code;
#endif

#if !defined(TLANG_GPU)
#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif
#endif

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace Tlang {
#define TLANG_NAMESPACE_END \
  }                         \
  }

#include <atomic>
#include <numeric>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <sys/time.h>

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
#if defined(TLANG_WITH_OPENMP)
#include <omp.h>
#endif

using float32 = float;
using float64 = double;
using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;
using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

#if defined(TLANG_GPU)
#include <cuda_runtime.h>
#undef FUNC_DECL
#define FUNC_DECL __host__ __device__
#endif

__device__ __host__ void exit() {
#if __CUDA_ARCH__
  assert(0);
#else
  exit(-1);
#endif
}

#if defined(TL_DEBUG)
#define TC_ASSERT(x)                                                     \
  if (!(x)) {                                                            \
    printf("Assertion failed (%s@Ln %d): %s\n", __FILE__, __LINE__, #x); \
    exit();                                                              \
  }

#define TC_ASSERT_INFO(x, t)                                            \
  if (!(x)) {                                                           \
    printf("Assertion failed (%s@Ln %d): %s\n", __FILE__, __LINE__, t); \
    exit();                                                             \
  }
#else
#define TC_ASSERT(x)
#define TC_ASSERT_INFO(x, t)
#endif

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

TLANG_NAMESPACE_BEGIN

inline double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

using size_t = std::size_t;

constexpr int max_num_indices = 4;
constexpr int max_num_args = 8;

struct SNodeMeta {
  int indices[max_num_indices];
  int active;
  int start_loop;
  int end_loop;
  int _;
  void **snode_ptr;
  void *ptr;
};

struct AllocatorStat {
  int snode_id;
  size_t pool_size;
  size_t num_resident_blocks;
  size_t num_recycled_blocks;
  SNodeMeta *resident_metas;
};

constexpr int max_num_snodes = 1024;
constexpr int max_gpu_block_size = 1024;

template <typename T, typename G>
__device__ __host__ T union_cast(G g) {
  static_assert(sizeof(T) == sizeof(G), "");
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

template <typename T, typename G>
__device__ __host__ T union_cast_different_size(G g) {
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

TLANG_NAMESPACE_END
