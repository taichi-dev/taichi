#pragma once

#define FUNC_DECL

#include <immintrin.h>

#if !defined(TC_INCLUDED)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <array>
#include <vector>

using float32 = float;
using float64 = double;
using int32 = int;
using uint64 = unsigned long long;

#if defined(TLANG_GPU)
#include <cuda_runtime.h>
#undef FUNC_DECL
#define FUNC_DECL __host__ __device__
#endif

#endif

namespace taichi {
namespace Tlang {

struct Context {
  static constexpr int max_num_buffers = 16;
  static constexpr int max_num_parameters = 16;
  static constexpr int max_num_ranges = 16;
  using Buffer = void *;
  Buffer buffers[max_num_buffers];
  double parameters[max_num_parameters];
  uint64 ranges[max_num_ranges];

  Context() {
    std::memset(buffers, 0, sizeof(buffers));
    std::memset(parameters, 0, sizeof(parameters));
    std::memset(ranges, 0, sizeof(ranges));
  }

  Context(void *x, void *y, void *z, uint64 n) : Context() {
    buffers[0] = x;
    buffers[1] = y;
    buffers[2] = z;
    ranges[0] = n;
  }

  template <typename T>
  FUNC_DECL T *get_buffer(int i) {
    return reinterpret_cast<T *>(buffers[i]);
  }

  template <typename T>
  FUNC_DECL T &get_parameter(int i) {
    return *reinterpret_cast<T *>(&parameters[i]);
  }

  FUNC_DECL uint64 &get_range(int i) {
    return ranges[i];
  }
};

// Virtual Vectors

template <int dim, typename T>
struct VV {  // Virtual Vector
  T d[dim];

  VV() {
  }

  VV(T v) {
    for (int i = 0; i < dim; i++) {
      d[i] = v;
    }
  }

  VV(std::array<T, dim> val) {
    for (int i = 0; i < dim; i++) {
      d[i] = val[i];
    }
  }

  T &operator[](int i) {
    return d[i];
  }

  const T &operator[](int i) const {
    return d[i];
  }

  void print() const {
    std::cout << "[";
    for (int i = 0; i < dim; i++) {
      std::cout << d[i] << ", ";
    }
    std::cout << "]\n";
  }
};

#define BINARY_OP(OPNAME, OP)                                               \
  template <int dim, typename T>                                            \
  inline VV<dim, T> operator OP(const VV<dim, T> &a, const VV<dim, T> &b) { \
    VV<dim, T> c;                                                           \
    for (int i = 0; i < dim; i++) {                                         \
      c[i] = a[i] OP b[i];                                                  \
    }                                                                       \
    return c;                                                               \
  }

BINARY_OP(add, +);
BINARY_OP(sub, -);
BINARY_OP(mul, *);
BINARY_OP(div, /);
BINARY_OP(mod, %);

#undef BINARY_OP

template <int dim, typename T>
inline VV<dim, T> min(const VV<dim, T> &a, const VV<dim, T> &b) {
  VV<dim, T> c;
  for (int i = 0; i < dim; i++) {
    c[i] = std::min(a[i], b[i]);
  }
  return c;
}

template <int dim, typename T>
inline VV<dim, T> max(const VV<dim, T> &a, const VV<dim, T> &b) {
  VV<dim, T> c;
  for (int i = 0; i < dim; i++) {
    c[i] = std::max(a[i], b[i]);
  }
  return c;
}

template <int dim, typename T>
inline VV<dim, T> floor(const VV<dim, T> &a) {
  VV<dim, T> c;
  for (int i = 0; i < dim; i++) {
    c[i] = std::floor(a[i]);
  }
  return c;
}

template <typename T, typename G, int dim>
inline VV<dim, T> cast(const VV<dim, G> &a) {
  VV<dim, T> c;
  for (int i = 0; i < dim; i++) {
    c[i] = (T)a[i];
  }
  return c;
}

template <int dim, typename T>
inline VV<dim, T> load(T *base_address, VV<dim, int> offsets) {
  VV<dim, T> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = *(base_address + offsets[i]);
  }
  return ret;
}

template <int dim, typename T>
inline void store(VV<dim, T> a, T *base_address, VV<dim, int> offsets) {
  for (int i = 0; i < dim; i++) {
    *(base_address + offsets[i]) = a[i];
  }
}

template <typename SA, int output_dim, typename T = typename SA::T>
inline VV<output_dim, T> shuffle(SA &a, VV<output_dim, int> offsets) {
  VV<output_dim, T> ret;
  for (int i = 0; i < output_dim; i++) {
    ret[i] = a[offsets[i]];
  }
  return ret;
};

template <typename T_,
          int num_groups,
          int num_inputs,
          int input_group_size,
          int output_group_size>
struct SlowAdapter {
  // static constexpr int num_outputs = num_inputs * input_group_size /
  // output_group_size;
  // static_assert(num_inputs * input_group_size % output_group_size == 0, "");

  using T = T_;
  // static constexpr int num_inputs = 8;
  static constexpr int num_outputs = 8;

  static constexpr int input_dim = num_groups * input_group_size;
  static constexpr int output_dim = num_groups * output_group_size;

  VV<input_dim, T> inputs[num_inputs];
  VV<output_dim, T> outputs[num_outputs];

  template <int i>
  void set(const VV<input_dim, T> &v) {
    static_assert(0 <= i && i < num_inputs, "");
    inputs[i] = v;
  }

  void set(int i, const VV<input_dim, T> &v) {
    inputs[i] = v;
  }

  void shuffle() {
    /*
    constexpr int num_elements = num_inputs * input_group_size;
    static_assert(num_inputs * num_elements * input_group_size ==
                      num_outputs * num_elements * output_group_size,
                  "");

    for (int i = 0; i < num_elements; i++) {
      for (int j = 0; j < num_groups; j++) {
        auto v = inputs[i / input_group_size]
                       [j * input_group_size + i % input_group_size];
        outputs[i / output_group_size]
               [j * output_group_size + i % output_group_size] = v;
      }
    }
    */
  }

  template <int i>
  auto get_input() {
    static_assert(0 <= i && i < num_inputs, "");
    return inputs[i];
  }

  template <int i>
  auto get() {
    static_assert(0 <= i && i < num_outputs, "");
    return outputs[i];
  }

  auto get(int i) {
    return outputs[i];
  }

  inline T &operator[](int i) {
    return inputs[i / input_dim][i % input_dim];
  }
};

// End Virtual Vectors

template <typename T, int dim>
struct vec_helper;

#define REGISTER_VEC(T, dim, name) \
  template <>                      \
  struct vec_helper<T, dim> {      \
    using type = name;             \
  };

REGISTER_VEC(float32, 8, __m256);
REGISTER_VEC(int32, 8, __m256i);
// REGISTER_VEC(uint32, 8, __m256u);

template <typename T, int dim>
using vec = typename vec_helper<T, dim>::type;

template <typename T, int dim>
inline vec<T, dim> load(const void *);

template <typename T, int dim>
inline vec<T, dim> floor(const vec<T, dim> &);

using float32x8 = vec<float32, 8>;

template <>
inline float32x8 floor<float32, 8>(const float32x8 &v) {
  return _mm256_floor_ps(v);
};

template <>
inline float32x8 load<float32, 8>(const void *addr) {
  return _mm256_load_ps((float32 *)addr);
};

template <>
inline vec<int32, 8> load<int32, 8>(const void *addr) {
  return _mm256_load_si256((__m256i *)addr);
};

template <typename T, int dim, int n>
struct vvec {
  vec<T, dim> d[n];
};

#define VVEC_BINARY_OP(NAME, OP)                          \
  template <typename T, int dim, int n>                   \
  inline vvec<T, dim, n> NAME(const vvec<T, dim, n> &a,   \
                              const vvec<T, dim, n> &b) { \
    vvec<T, dim, n> ret;                                  \
    for (int i = 0; i < n; i++) {                         \
      ret.d[i] = a.d[i] OP b.d[i];                        \
    }                                                     \
    return ret;                                           \
  }

VVEC_BINARY_OP(add, +);
VVEC_BINARY_OP(sub, -);
VVEC_BINARY_OP(mul, *);
VVEC_BINARY_OP(div, /);
VVEC_BINARY_OP(mod, %);
}
}
