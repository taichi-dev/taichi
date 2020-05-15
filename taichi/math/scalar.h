/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::cos;
using std::floor;
using std::max;
using std::min;
using std::sin;
using std::tan;

const real pi{acosf(-1.0_f)};
const real eps = 1e-6_f;

template <int I, typename T>
constexpr TI_FORCE_INLINE T pow(T a) noexcept {
  T ret(1);
  for (int i = 0; i < I; i++) {
    ret *= a;
  }
  return ret;
};

TI_FORCE_INLINE float32 fract(float32 a) noexcept {
  return a - (int)floor(a);
}

TI_FORCE_INLINE float64 fract(float64 a) noexcept {
  return a - (int)floor(a);
}

template <typename T>
TI_FORCE_INLINE T clamp(const T &a, const T &min, const T &max) noexcept {
  if (a < min)
    return min;
  if (a > max)
    return max;
  return a;
}

template <typename T>
TI_FORCE_INLINE T clamp01(const T &a) noexcept {
  if (a < T(0))
    return T(0);
  if (a > T(1))
    return T(1);
  return a;
}

template <typename T>
TI_FORCE_INLINE T clamp(const T &a) noexcept {
  return clamp01(a);
}

template <typename T, typename V>
TI_FORCE_INLINE V lerp(T a, V x_0, V x_1) noexcept {
  return V((T(1) - a) * x_0 + a * x_1);
}

template <typename T>
TI_FORCE_INLINE T sqr(const T &a) noexcept {
  return pow<2>(a);
}

TI_FORCE_INLINE int sgn(float a) noexcept {
  if (a < -eps)
    return -1;
  else if (a > eps)
    return 1;
  return 0;
}

TI_FORCE_INLINE int sgn(double a) noexcept {
  if (a < -eps)
    return -1;
  else if (a > eps)
    return 1;
  return 0;
}

TI_FORCE_INLINE uint32 rand_int() noexcept {
  static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
  unsigned int t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

TI_FORCE_INLINE uint64 rand_int64() noexcept {
  return ((uint64)rand_int() << 32) + rand_int();
}

// inline float frand() { return (float)rand() / (RAND_MAX + 1); }
TI_FORCE_INLINE float32 rand() noexcept {
  return rand_int() * (1.0_f / 4294967296.0f);
}

template <typename T>
TI_FORCE_INLINE T rand() noexcept;

template <>
TI_FORCE_INLINE float rand<float>() noexcept {
  return rand_int() * (1.0_f / 4294967296.0f);
}

template <>
TI_FORCE_INLINE double rand<double>() noexcept {
  return rand_int() * (1.0 / 4294967296.0);
}

template <>
TI_FORCE_INLINE int rand<int>() noexcept {
  return rand_int();
}

inline int is_prime(int a) noexcept {
  assert(a >= 2);
  for (int i = 2; i * i <= a; i++) {
    if (a % i == 0)
      return false;
  }
  return true;
}

template <typename T>
TI_FORCE_INLINE T hypot2(const T &x, const T &y) noexcept {
  return x * x + y * y;
}

TI_FORCE_INLINE float32 pow(const float32 &a, const float32 &b) noexcept {
  return ::pow(a, b);
}

TI_FORCE_INLINE float64 pow(const float64 &a, const float64 &b) noexcept {
  return ::pow(a, b);
}

template <typename T>
TI_FORCE_INLINE bool is_normal(T m) noexcept {
  return std::isfinite(m);
}

template <typename T>
TI_FORCE_INLINE bool abnormal(T m) noexcept {
  return !is_normal(m);
}

inline int64 get_largest_pot(int64 a) noexcept {
  TI_ASSERT_INFO(a > 0,
                 "a should be positive, instead of " + std::to_string(a));
  // TODO: optimize
  int64 i = 1;
  while (i <= a / 2) {
    i *= 2;
  }
  return i;
}

TI_NAMESPACE_END
