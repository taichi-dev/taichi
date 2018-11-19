#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math.h>
#include <taichi/system/timer.h>
#include "tlang.h"
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

constexpr int rounds = 16384 * 20;
constexpr int N = 512;

template <int dim, typename T>
real eigen_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int i = 0; i < N; i++) {
      C[i] = A[i] * B[i];
    }
  }
  return Time::get_time() - t;
};

template <int dim, typename T>
real taichi_matmatmul() {
  std::vector<TMatrix<T, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int i = 0; i < N; i++) {
      C[i] = A[i] * B[i];
    }
  }
  return Time::get_time() - t;
}

template <int dim, typename T>
real AOS_matmatmul() {
  struct Mat {
    T d[dim][dim];
  };
  std::vector<Mat> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N; t++) {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          T sum = 0;
          for (int k = 0; k < dim; k++) {
            sum += A[t].d[i][k] * B[t].d[k][j];
          }
          C[t].d[i][j] = sum;
        }
      }
    }
  }
  return Time::get_time() - t;
}

template <int dim, typename T>
real AOS2_matmatmul() {
  struct Mat {
    T d[dim][dim];
  };
  std::vector<Mat> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N; t++) {
      Mat X, Y;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          X.d[i][j] = A[t].d[i][j];
          Y.d[i][j] = B[t].d[i][j];
        }
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          T sum = 0;
          for (int k = 0; k < dim; k++) {
            sum += X.d[i][k] * Y.d[k][j];
          }
          C[t].d[i][j] = sum;
        }
      }
    }
  }
  return Time::get_time() - t;
}

class AlignedAllocator {
  std::vector<uint8> _data;
  void *data;

 public:
  AlignedAllocator(std::size_t size) {
    _data.resize(size + 4096);
    auto p = reinterpret_cast<uint64>(_data.data());
    data = (void *)(p + (4096 - p % 4096));
  }

  template <typename T>
  T *get() {
    return reinterpret_cast<T *>(data);
  }
};

// array of N * dim * dim * 8 * float32
template <int dim>
void AOSOA_matmul(float32 *A, float32 *B, float32 *C) {
  constexpr int simd_width = 8;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256 a[dim * dim], b[dim * dim];
      const int p = dim * dim * simd_width * t;
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_ps(&A[p + simd_width * i]);
        b[i] = _mm256_load_ps(&B[p + simd_width * i]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256 c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_ps(&C[p + simd_width * (i * dim + j)], c);
        }
      }
    }
  }
}

// array of N * dim * dim * 8 * float64
template <int dim>
void AOSOA_matmul(float64 *A, float64 *B, float64 *C) {
  constexpr int simd_width = 4;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256d a[dim * dim], b[dim * dim];
      const int p = dim * dim * simd_width * t;
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_pd(&A[p + simd_width * i]);
        b[i] = _mm256_load_pd(&B[p + simd_width * i]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256d c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_pd(&C[p + simd_width * (i * dim + j)], c);
        }
      }
    }
  }
}

template <int dim, typename T>
real AOSOA_matmatmul() {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  auto t = Time::get_time();
  AOSOA_matmul<dim>(A.get<T>(), B.get<T>(), C.get<T>());
  return Time::get_time() - t;
};

// array of N * dim * dim * 8 * float32
template <int dim>
void SOA_matmul(float32 *A, float32 *B, float32 *C) {
  constexpr int simd_width = 8;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256 a[dim * dim], b[dim * dim];
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_ps(&A[i * N + t * simd_width]);
        b[i] = _mm256_load_ps(&B[i * N + t * simd_width]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256 c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_ps(&C[(i * dim + j) * N + t * simd_width], c);
        }
      }
    }
  }
}

// array of N * dim * dim * 8 * float64
template <int dim>
void SOA_matmul(float64 *A, float64 *B, float64 *C) {
  constexpr int simd_width = 4;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256d a[dim * dim], b[dim * dim];
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_pd(&A[i * N + t * simd_width]);
        b[i] = _mm256_load_pd(&B[i * N + t * simd_width]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256d c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_pd(&C[(i * dim + j) * N + t * simd_width], c);
        }
      }
    }
  }
}

template <int dim, typename T>
real SOA_matmatmul() {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  auto t = Time::get_time();
  SOA_matmul<dim>(A.get<T>(), B.get<T>(), C.get<T>());
  return Time::get_time() - t;
};

template <int dim, typename T>
real Tlang_matmatmul() {
  using namespace Tlang;

  Expr a[dim][dim], b[dim][dim];

  int simd_width = 32 / sizeof(float32);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a[i][j] = load(0, dim * dim, simd_width * (i * dim + j));
      b[i][j] = load(1, dim * dim, simd_width * (i * dim + j));
    }
  }

  Expr ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      auto sum = a[i][0] * b[0][j];
      for (int k = 1; k < dim; k++) {
        sum = sum + a[i][k] * b[k][j];
      }
      ret.store(sum, 2, dim * dim, simd_width * (i * dim + j));
    }
  }

  CodeGen cg;
  auto func = cg.get(ret);

  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);
  AlignedAllocator D(sizeof(T) * N * dim * dim);

  for (int i = 0; i < N * dim * dim; i++) {
    A.get<T>()[i] = rand();
    B.get<T>()[i] = rand();
  }

  auto t = Time::get_time();
  for (int i = 0; i < rounds; i++) {
    func(A.get<T>(), B.get<T>(), C.get<T>(), N);
  }
  t = Time::get_time() - t;

  AOSOA_matmul<dim>(A.get<T>(), B.get<T>(), D.get<T>());

  for (int i = 0; i < N * dim * dim; i++) {
    auto a = C.get<T>()[i];
    auto b = D.get<T>()[i];
    TC_ASSERT(std::abs(a - b) < 1e-5_f);
  }

  return t;
}

#define BENCHMARK(x)                                                          \
  {                                                                           \
    real t = x##_matmatmul<dim, T>();                                         \
    fmt::print("Matrix<{}, {}>    {:8s} = {:8.3f} ms  {:8.3f} cyc / elem \n", \
               dim, sizeof(T) == 4 ? "float32" : "float64", #x, t * 1000.0_f, \
               4.2 * 1e9 * t / rounds / N);                                   \
  }

template <int dim, typename T>
void run() {
  BENCHMARK(eigen);
  BENCHMARK(taichi);
  BENCHMARK(AOS);
  BENCHMARK(AOS2);
  BENCHMARK(SOA);
  BENCHMARK(AOSOA);
  BENCHMARK(Tlang);
  fmt::print("\n");
}

auto benchmark_matmul = []() {
  run<2, float32>();
  run<3, float32>();
  run<4, float32>();
  /*
  run<2, float64>();
  run<3, float64>();
  run<4, float64>();
  */
};
TC_REGISTER_TASK(benchmark_matmul);

TC_NAMESPACE_END
