#pragma once

#define FUNC_DECL

#if !defined(TC_INCLUDED)

#include <cstdio>
#include <cstdlib>
#include <cstring>
using float32 = float;
using float64 = double;
using uint64 = unsigned long long;

#if defined(TLANG_CPU)
#include <immintrin.h>
#endif

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
struct VV { // Virtual Vector
  T d[dim];

  T &operator[](int i) {
    return d[i];
  }
};

#define BINARY_OP(OPNAME, OP) template<int dim, typename T> \
inline VV<dim, T> OPNAME(const VV<dim, T> &a, const VV<dim, T> &b) { \
  VV<dim, T> c; \
  for (int i = 0; i < dim; i++) { \
    c[i] = a[i] OP b[i]; \
  } \
}

BINARY_OP(add, +);
BINARY_OP(sub, -);
BINARY_OP(mul, *);
BINARY_OP(div, /);

#undef BINARY_OP

template <int dim, typename T>
inline VV<dim, T> load(T *base_address, VV<dim, int> offsets) {
  VV<dim, T> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = *(base_address + offsets[i]);
  }
  return ret;
};

template <int dim, typename T>
inline void store(VV<dim, T> a, T *base_address, VV<dim, int> offsets) {
  for (int i = 0; i < dim; i++) {
    *(base_address + offsets[i]) = a[i];
  }
};

// TODO: adapters


// End Virtual Vectors

}
}
