#include <fstream>
#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/system/timer.h>
#include <Eigen/StdVector>
#include "tlang.h"
// #include <taichi/testing.h>
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T>>;

constexpr real cpu_frequency = 3.6_f;

// constexpr int enlarge = 4096;
constexpr int enlarge = 1;
constexpr int rounds = 16384 * 2048 / enlarge;
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

  template <typename T = void>
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

  Matrix(int n, int m = 1) : n(n), m(m) {
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

void test_vec_add() {
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
  auto func = cg.get(ret, 1);

  TC_ALIGNED(64) float32 x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    x[i] = i;
    y[i] = -2 * i;
  }
  func(x, y, z, n);
  for (int i = 0; i < n; i++) {
    TC_ASSERT(z[i] == -i);
  }
}

void print_time(float64 t, int64 elements) {
  fmt::print("   {:10.3f} cyc / elem      [adjusted run time = {:10.3f} ms] \n",
             cpu_frequency * 1e9 * t / elements, t * 1000.0_f);
}

template <int dim>
void test_mat_vec_mul_eigen(bool in_cache) {
  fmt::print("dim={} Eigen in_cache={}\n", dim, in_cache);
  using namespace Tlang;

  int enlarge = in_cache ? 1 : 4096;
  int64 n = taichi::N * enlarge;
  int64 rounds = taichi::rounds / enlarge / dim / dim / (in_cache ? 1 : 5) / 2;

  EigenVector<Eigen::Matrix<float32, dim, dim>> m(n);
  EigenVector<Eigen::Matrix<float32, dim, 1>> v(n);
  EigenVector<Eigen::Matrix<float32, dim, 1>> mv(n);

  for (int K = 0; K < 1; K++) {
    float64 t = Time::get_time();
    for (int i = 0; i < rounds; i++) {
      for (int i = 0; i < n; i++) {
        mv[i] = m[i] * v[i];
      }
    }
    taichi::trash(mv[5]);
    print_time(Time::get_time() - t, n * rounds);
  }
}

template <int dim>
void test_mat_vec_mul(bool aosoa, bool in_cache, int unroll, int prefetch) {
  fmt::print("dim={} {} in_cache={} unroll={} prefetch={}\n", dim,
             aosoa ? "aosoa" : "soa", (int)in_cache, unroll, prefetch);
  using namespace Tlang;
  constexpr int simd_width = 8;
  Matrix m(dim, dim), v(dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Address addr;
      addr.stream_id = 0;
      if (aosoa) {
        addr.coeff_i = dim;
        addr.coeff_aosoa_group_size = simd_width / dim;
        addr.coeff_aosoa_stride = simd_width * (dim - 1);
        addr.coeff_const = j * simd_width + i;
      } else {
        addr.coeff_i = 1;
        addr.coeff_imax = i * dim + j;
      }
      m(i, j) = load(addr);
    }
    Address addr;
    addr.stream_id = 1;
    if (aosoa) {
      addr.coeff_i = dim;
      addr.coeff_const = i;
    } else {
      addr.coeff_i = 1;
      addr.coeff_imax = i;
    }
    v(i) = load(addr);
  }
  Expr ret;
  auto mv = m * v;
  for (int i = 0; i < dim; i++) {
    Address addr;
    addr.stream_id = 2;
    if (aosoa) {
      addr.coeff_i = dim;
      addr.coeff_const = i;
    } else {
      addr.coeff_i = 1;
      addr.coeff_imax = i;
    }
    mv(i) = ret.store(mv(i), addr);
  }

  int64 enlarge = in_cache ? 1 : 4096;
  int64 n = taichi::N * enlarge;
  int64 rounds = taichi::rounds / enlarge / dim / dim / (in_cache ? 1 : 5);
  CodeGen cg;
  cg.unroll = unroll;
  cg.prefetch = prefetch;
  TC_ASSERT(8 % dim == 0);
  auto func = cg.get(ret, aosoa ? dim : 1);

  AlignedAllocator M_allocator(dim * dim * n * sizeof(float32)),
      V_allocator(dim * n * sizeof(float32)),
      MV_allocator(dim * n * sizeof(float32));

  std::vector<Eigen::Matrix<float32, dim, 1>> ground_truth(n);
  for (int i = 0; i < n; i++) {
    Eigen::Matrix<float32, dim, dim> m_gt;
    Eigen::Matrix<float32, dim, 1> v_gt;
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        m_gt(j, k) = rand<float32>();
        M_allocator.get<float32>()[m(j, k)->addr.eval(i, n)] = m_gt(j, k);
      }
      v_gt(j) = rand<float32>();
      V_allocator.get<float32>()[v(j)->addr.eval(i, n)] = v_gt(j);
    }
    Eigen::Matrix<float32, dim, 1> mv_gt = m_gt * v_gt;
    ground_truth[i] = mv_gt;
  }

  // warm up
  for (int i = 0; i < 10; i++)
    func(M_allocator.get<float32>(), V_allocator.get<float32>(),
         MV_allocator.get<float32>(), n);

  for (int K = 0; K < 1; K++) {
    float64 t = Time::get_time();
    for (int i = 0; i < rounds; i++) {
      func(M_allocator.get<float32>(), V_allocator.get<float32>(),
           MV_allocator.get<float32>(), n);
    }
    print_time(Time::get_time() - t, n * rounds);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      auto computed = MV_allocator.get<float32>()[mv(j)->addr.eval(i, n)];
      auto gt = ground_truth[i](j);
      if (std::abs(computed - gt) > 1e-4_f) {
        TC_P(i);
        TC_P(j);
        TC_P(computed);
        TC_P(gt);
        TC_ERROR("Failed!");
      }
    }
  }
}

template <int dim>
void test_mat_vec_mul_all() {
  for (auto in_cache : {false, true}) {
    test_mat_vec_mul_eigen<dim>(in_cache);
    for (auto aosoa : {false, true}) {
      for (auto unroll : {1, 4, 16})
        for (auto prefetch : {0, 16, 64})
          test_mat_vec_mul<dim>(aosoa, in_cache, unroll, prefetch);
    }
    fmt::print("\n");
  }
}

auto test_tlang = []() {
  test_mat_vec_mul_all<1>();
  test_mat_vec_mul_all<2>();
  test_mat_vec_mul_all<4>();
  test_vec_add();
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

void memcpy_intel(void *a_, void *b_, std::size_t size) {
  constexpr int PAGESIZE = 4096;
  constexpr int NUMPERPAGE = 512;  // # of elements to fit a page
  std::size_t N = size / 8;
  double *a = (double *)a_;
  double *b = (double *)b_;
  double temp;
  for (int kk = 0; kk < N; kk += NUMPERPAGE) {
    temp = a[kk + NUMPERPAGE];  // TLB priming
    trash(temp);
    // use block size = page size,
    // prefetch entire block, one cache line per loop
    for (int j = kk + 16; j < kk + NUMPERPAGE; j += 16) {
      _mm_prefetch((char *)&a[j], _MM_HINT_NTA);
    }
    // copy 128 byte per loop
    for (int j = kk; j < kk + NUMPERPAGE; j += 16) {
      _mm_stream_ps((float *)&b[j], _mm_load_ps((float *)&a[j]));
      _mm_stream_ps((float *)&b[j + 2], _mm_load_ps((float *)&a[j + 2]));
      _mm_stream_ps((float *)&b[j + 4], _mm_load_ps((float *)&a[j + 4]));
      _mm_stream_ps((float *)&b[j + 6], _mm_load_ps((float *)&a[j + 6]));
      _mm_stream_ps((float *)&b[j + 8], _mm_load_ps((float *)&a[j + 8]));
      _mm_stream_ps((float *)&b[j + 10], _mm_load_ps((float *)&a[j + 10]));
      _mm_stream_ps((float *)&b[j + 12], _mm_load_ps((float *)&a[j + 12]));
      _mm_stream_ps((float *)&b[j + 14], _mm_load_ps((float *)&a[j + 14]));
    }  // finished copying one block
  }    // finished copying N elements
  _mm_sfence();
}

auto memcpy_test = []() {
  auto size = 1024 * 1024 * 1024;
  AlignedAllocator a(size), b(size);

  int repeat = 100;
  float64 t;

  t = Time::get_time();
  for (int i = 0; i < repeat; i++) {
    memcpy(a.get(), b.get(), size);
  }
  TC_P(Time::get_time() - t);

  t = Time::get_time();
  for (int i = 0; i < repeat; i++) {
    memset(a.get(), 0, size);
  }
  TC_P(Time::get_time() - t);

  t = Time::get_time();
  for (int i = 0; i < repeat; i++) {
    memcpy_intel(a.get(), b.get(), size);
  }
  TC_P(Time::get_time() - t);

  t = Time::get_time();
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < size / 8; j++) {
      a.get<double>()[j] = b.get<double>()[j];
    }
  }
  TC_P(Time::get_time() - t);
};

TC_REGISTER_TASK(memcpy_test);

auto allocator_test  = []() {
  using namespace Tlang;
  MemoryAllocator alloc;
};

TC_REGISTER_TASK(allocator_test);

TC_NAMESPACE_END
