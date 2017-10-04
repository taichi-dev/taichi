/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

#undef max
#undef min

using std::min;
using std::max;
using std::abs;
using std::sin;
using std::asin;
using std::cos;
using std::acos;
using std::tan;
using std::atan;
using std::floor;

const real pi{acosf(-1.0_f)};
const real eps = 1e-6_f;

template <int I, typename T>
constexpr inline T pow(T a) {
  T ret(1);
  for (int i = 0; i < I; i++) {
    ret *= a;
  }
  return ret;
};

inline float32 fract(float32 a) {
  return a - (int)floor(a);
}

inline float64 fract(float64 a) {
  return a - (int)floor(a);
}

template <typename T>
inline T clamp(const T &a, const T &min, const T &max) {
  if (a < min)
    return min;
  if (a > max)
    return max;
  return a;
}

template <typename T>
inline T clamp(const T &a) {
  if (a < T(0))
    return T(0);
  if (a > T(1))
    return T(1);
  return a;
}

template <typename T, typename V>
inline V lerp(T a, V x_0, V x_1) {
  return (T(1) - a) * x_0 + a * x_1;
}

template <typename T>
T sqr(const T &a) {
  return pow<2>(a);
}

inline int sgn(float a) {
  if (a < -eps)
    return -1;
  else if (a > eps)
    return 1;
  return 0;
}

inline int sgn(double a) {
  if (a < -eps)
    return -1;
  else if (a > eps)
    return 1;
  return 0;
}

inline uint32 rand_int() {
  static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
  unsigned int t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

// inline float frand() { return (float)rand() / (RAND_MAX + 1); }
inline float rand() {
  return rand_int() * (1.0_f / 4294967296.0f);
}

inline int is_prime(int a) {
  assert(a >= 2);
  for (int i = 2; i * i <= a; i++) {
    if (a % i == 0)
      return false;
  }
  return true;
}

template <typename T>
inline T hypot2(const T &x, const T &y) {
  return x * x + y * y;
}

inline float32 pow(const float32 &a, const float32 &b) {
  return ::pow(a, b);
}

inline float64 pow(const float64 &a, const float64 &b) {
  return ::pow(a, b);
}

template <typename T>
inline bool is_normal(T m) {
  return std::isfinite(m);
}

template <typename T>
inline bool abnormal(T m) {
  return !is_normal(m);
}

inline int64 get_largest_pot(int64 a) {
  assert_info(a > 0, "a should be positive, instead of " + std::to_string(a));
  // TODO: optimize
  int64 i = 1;
  while (i <= a / 2) {
    i *= 2;
  }
  return i;
}

TC_NAMESPACE_END
