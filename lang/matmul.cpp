#include <fstream>
#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/system/timer.h>
#include "tlang.h"
// #include <taichi/testing.h>
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

constexpr real cpu_frequency = 4.2_f;

// constexpr int enlarge = 4096;
constexpr int enlarge = 1;
constexpr int rounds = 16384 * 20 / enlarge;
constexpr int N = 256 * enlarge;

template <int dim, typename T>
real AOS_eigen_matmatmul() {
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
real AOS_eigen_unroll2_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int i = 0; i < N; i += 2) {
      C[i] = A[i] * B[i];
      C[i + 1] = A[i + 1] * B[i + 1];
    }
  }
  return Time::get_time() - t;
};

template <int dim, typename T>
real AOS_eigen_unroll4_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int i = 0; i < N; i += 4) {
      C[i] = A[i] * B[i];
      C[i + 1] = A[i + 1] * B[i + 1];
      C[i + 2] = A[i + 2] * B[i + 2];
      C[i + 3] = A[i + 3] * B[i + 3];
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
real AOSOA_AVX2_matmatmul() {
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
real SOA_AVX2_matmatmul() {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  auto t = Time::get_time();
  SOA_matmul<dim>(A.get<T>(), B.get<T>(), C.get<T>());
  return Time::get_time() - t;
};

namespace Tlang {
struct Matrix {
  int n, m;
  std::vector<Expr> entries;

  Matrix(int n, int m) : n(n), m(m) {
    entries.resize(n * m, Expr());
  }

  Expr &operator()(int i, int j) {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  const Expr &operator()(int i, int j) const {
    TC_ASSERT(0 <= i && i < n);
    TC_ASSERT(0 <= j && j < n);
    return entries[i * m + j];
  }

  Expr &operator()(int i) {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }

  const Expr &operator()(int i) const {
    TC_ASSERT(0 <= i && i < n * m);
    TC_ASSERT(n == 1 || m == 1);
    return entries[i];
  }
};

Matrix operator*(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.m == B.n);
  Matrix C(A.n, B.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < B.m; j++) {
      C(i, j) = A(i, 0) * B(0, j);
      for (int k = 1; k < A.m; k++) {
        C(i, j) = C(i, j) + A(i, k) * B(k, j);
      }
    }
  }
  return C;
}

Matrix operator+(const Matrix &A, const Matrix &B) {
  TC_ASSERT(A.n == B.n);
  TC_ASSERT(A.m == B.m);
  Matrix C(A.n, A.m);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.m; j++) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
}

}  // namespace Tlang

template <int dim, typename T>
real Tlang_matmatmul(Tlang::CodeGen::Mode mode, int simd_width) {
  using namespace Tlang;

  Matrix a(dim, dim), b(dim, dim);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Address addr;
      addr.stream_id = 0;
      addr.coeff_i = 1;
      addr.coeff_aosoa_stride = simd_width * dim * dim;
      addr.coeff_aosoa_group_size = simd_width;
      addr.coeff_const = simd_width * (i * dim + j);
      a(i, j) = load(addr);
      addr.stream_id = 1;
      b(i, j) = load(addr);
    }
  }

  auto c = a * b;

  Expr ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Address addr;
      addr.stream_id = 2;
      addr.coeff_i = dim * dim;
      addr.coeff_const = simd_width * (i * dim + j);
      ret.store(c(i, j), addr);
    }
  }

  CodeGen cg(mode, simd_width);
  auto func = cg.get(ret, dim);

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

  if (simd_width == 8) {
    AOSOA_matmul<dim>(A.get<T>(), B.get<T>(), D.get<T>());

    for (int i = 0; i < N * dim * dim; i++) {
      auto a = C.get<T>()[i];
      auto b = D.get<T>()[i];
      TC_ASSERT(std::abs(a - b) < 1e-5_f);
    }
  }

  return t;
}

template <int dim, typename T>
real TlangVec8_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CodeGen::Mode::vector, 8);
}

template <int dim, typename T>
real TlangSca8_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CodeGen::Mode::scalar, 8);
}

template <int dim, typename T>
real TlangVec16_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CodeGen::Mode::vector, 16);
}

template <int dim, typename T>
real TlangSca16_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CodeGen::Mode::scalar, 16);
}

#define BENCHMARK(x)                                                 \
  {                                                                  \
    real t = x##_matmatmul<dim, T>();                                \
    fmt::print("  {:18s} = {:10.3f} ms  {:10.3f} cyc / elem \n", #x, \
               t * 1000.0_f, cpu_frequency * 1e9 * t / rounds / N);  \
  }

template <int dim, typename T>
void run() {
  fmt::print("Matrix<{}, {}>:\n", dim, sizeof(T) == 4 ? "float32" : "float64");
  BENCHMARK(TlangVec8);
  // BENCHMARK(TlangSca8);
  // BENCHMARK(TlangVec16);
  // BENCHMARK(TlangSca16);
  BENCHMARK(AOS_eigen);
  BENCHMARK(AOS_eigen_unroll2);
  BENCHMARK(AOS_eigen_unroll4);
  // BENCHMARK(taichi);
  // BENCHMARK(AOS);
  // BENCHMARK(AOS2);
  BENCHMARK(SOA_AVX2);
  BENCHMARK(AOSOA_AVX2);
  fmt::print("\n");
}

auto benchmark_matmul = []() {
#if defined(TC_PLATFORM_LINUX)
  std::ifstream noturbo("/sys/devices/system/cpu/intel_pstate/no_turbo");
  char c;
  noturbo >> c;
  TC_WARN_IF(c != '1',
             "You seem to be running the benchmark with Intel Turboboost.");
#endif
  TC_INFO("Eigen Version {}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
          EIGEN_MINOR_VERSION);
  TC_INFO("GCC   Version {}.{}.{}", __GNUC__, __GNUC_MINOR__,
          __GNUC_PATCHLEVEL__);

  run<2, float32>();
  run<3, float32>();
  run<4, float32>();
  run<5, float32>();
  /*
  run<6, float32>();
  run<7, float32>();
  run<8, float32>();
  run<9, float32>();
  run<10, float32>();
  run<2, float64>();
  run<3, float64>();
  run<4, float64>();
  */
};
TC_REGISTER_TASK(benchmark_matmul);

/*
TC_TEST("Address allocation") {
  TC_CHECK(1 + 1 == 2);
}
*/

auto test_tlang = []() {
  using namespace Tlang;
  constexpr int n = 16;
  Address addr;
  addr.stream_id = 0;
  addr.coeff_i = 1;
  Expr a = load(addr);
  addr.stream_id = 1;
  Expr b = load(addr);
  auto c = a + b;
  Expr ret;
  addr.stream_id = 2;
  ret.store(c, addr);
  CodeGen cg;
  auto func = cg.get(ret, 8);

  TC_ALIGNED(64) float32 x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    x[i] = i;
    y[i] = -2 * i;
  }
  func(x, y, z, n);
  for (int i = 0; i < n; i++) {
    TC_INFO("z[{}] = {}", i, z[i]);
  }
};
TC_REGISTER_TASK(test_tlang);

auto test_slp = []() {
  using namespace Tlang;
  int M = 4;
  Matrix vec_a(M, 1);
  Matrix vec_b(M, 1);
  Expr ret;
  for (int i = 0; i < M; i++) {
    Address addr;
    addr.coeff_i = 1;
    addr.coeff_const = i;

    addr.stream_id = 0;
    vec_a(i) = load(addr);

    addr.stream_id = 1;
    vec_b(i) = load(addr);
  }
  Matrix vec_c = vec_a + vec_b;
  for (int i = 0; i < M; i++) {
    Address addr;
    addr.coeff_i = 1;
    addr.coeff_const = i;

    addr.stream_id = 2;
    ret.store(vec_c(i), addr);
  }

  CodeGen cg;
  auto func = cg.get(ret, 4);

  constexpr int n = 16;
  TC_ALIGNED(64) float32 x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    x[i] = i;
    y[i] = -2 * i;
  }
  func(x, y, z, n);
  for (int i = 0; i < n; i++) {
    TC_INFO("z[{}] = {}", i, z[i]);
  }
};

TC_REGISTER_TASK(test_slp)

TC_NAMESPACE_END
