#include <fstream>
#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/system/timer.h>
#include <Eigen/StdVector>
#include "tlang.h"
// #include <taichi/testing.h>
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

constexpr int simd_width = 8;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T>>;

constexpr real cpu_frequency = 4.2_f;

// constexpr int enlarge = 4096;
constexpr int enlarge = 1;
constexpr int rounds = 16384 / 8 * 2048 / enlarge;
constexpr int N = 256 * enlarge;

real default_measurement_time = 1;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time) {
  // first make rough estimate of run time.
  int64 batch_size = 1;
  while (true) {
    float64 t = Time::get_time();
    for (int64 i = 0; i < batch_size; i++) {
      target();
    }
    t = Time::get_time() - t;
    if (t < 0.05) {
      batch_size *= 2;
    } else {
      break;
    }
  }

  int64 total_batches = 0;
  float64 start_t = Time::get_time();
  while (Time::get_time() - start_t < time_second) {
    for (int i = 0; i < batch_size; i++) {
      target();
    }
    total_batches += batch_size;
  }
  return (Time::get_time() - start_t) / (total_batches * elements_per_call) *
         1e9_f64 * cpu_frequency;
}

template <int dim, typename T>
real AOS_eigen_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  return measure_cpe(
      [&]() {
        for (int i = 0; i < N; i++) {
          C[i] = A[i] * B[i];
        }
      },
      N);
};

template <int dim, typename T>
real AOS_eigen_unroll2_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  return measure_cpe(
      [&]() {
        for (int i = 0; i < N; i += 2) {
          C[i] = A[i] * B[i];
          C[i + 1] = A[i + 1] * B[i + 1];
        }
      },
      N);
};

template <int dim, typename T>
real AOS_eigen_unroll4_matmatmul() {
  std::vector<Eigen::Matrix<T, dim, dim>> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  return measure_cpe(
      [&]() {
        for (int i = 0; i < N; i += 4) {
          C[i] = A[i] * B[i];
          C[i + 1] = A[i + 1] * B[i + 1];
          C[i + 2] = A[i + 2] * B[i + 2];
          C[i + 3] = A[i + 3] * B[i + 3];
        }
      },
      N);
};

template <int dim, typename T>
real AOS_matmatmul() {
  struct Mat {
    T d[dim][dim];
  };
  std::vector<Mat> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  return measure_cpe(
      [&]() {
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
      },
      N);
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
real AOSOA_matmul(float32 *A, float32 *B, float32 *C) {
  constexpr int simd_width = 8;
  auto task = [&]() {
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
  };
  return measure_cpe(task, N);
}

// array of N * dim * dim * 8 * float64
/*
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
*/

template <int dim, typename T>
real AOSOA_AVX2_matmatmul() {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  return AOSOA_matmul<dim>(A.get<T>(), B.get<T>(), C.get<T>());
};

// array of N * dim * dim * 8 * float32
template <int dim>
real SOA_matmul(float32 *A, float32 *B, float32 *C) {
  constexpr int simd_width = 8;
  auto task = [&]() {
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
  };
  return measure_cpe(task, N);
}

/*
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
*/

template <int dim, typename T>
real SOA_AVX2_matmatmul() {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  return SOA_matmul<dim>(A.get<T>(), B.get<T>(), C.get<T>());
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
real Tlang_matmatmul(Tlang::CPUCodeGen::Mode mode,
                     int simd_width,
                     int layout = 0) {
  using namespace Tlang;

  Matrix a(dim, dim), b(dim, dim);

  CodeGen cg;
  auto &alloc = cg.alloc;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (layout == 0) {
        // AOSOA
        alloc.buffer(0)
            .stream(0)
            .group(0)
            .group(i * dim + j)
            .repeat(simd_width)
            .place(a(i, j));
        alloc.buffer(1)
            .stream(0)
            .group(0)
            .group(i * dim + j)
            .repeat(simd_width)
            .place(b(i, j));
      } else {
        alloc.buffer(0)
            .stream(0)
            .group(j)
            .repeat(simd_width / dim)
            .place(a(i, j));
        alloc.buffer(1)
            .stream(0)
            .group(j)
            .repeat(simd_width / dim)
            .place(b(i, j));
      }
    }
  }

  auto c = a * b;

  Expr ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (layout == 0) {
        c(i, j) = ret.store(c(i, j));
      } else {
        c(j, i) = ret.store(c(j, i));
      }
    }
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (layout == 0) {
        alloc.buffer(2)
            .stream(0)
            .group(0)
            .group(i * dim + j)
            .repeat(simd_width)
            .place(c(i, j));
      } else {
        alloc.buffer(2)
            .stream(0)
            .group(j)
            .repeat(simd_width / dim)
            .place(c(i, j));
      }
    }
  }

  auto func = cg.get(ret, layout == 0 ? 1 : dim, mode, simd_width);

  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);
  AlignedAllocator D(sizeof(T) * N * dim * dim);

  AlignedAllocator A_(sizeof(T) * N * dim * dim);
  AlignedAllocator B_(sizeof(T) * N * dim * dim);

  for (int i = 0; i < N * dim * dim; i++) {
    A.get<T>()[i] = rand();
    B.get<T>()[i] = rand();
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        int ind1 = c(j, k)->addr.eval(i, N);
        int ind2 = (i / simd_width * dim * dim + j * dim + k) * simd_width +
                   i % simd_width;
        A_.get<float32>()[ind1] = A.get<float32>()[ind2];
        B_.get<float32>()[ind1] = B.get<float32>()[ind2];
      }
    }
  }

  auto cpe =
      measure_cpe([&]() { func(A_.get<T>(), B_.get<T>(), C.get<T>(), N); }, N);

  if (simd_width == 8) {
    AOSOA_matmul<dim>(A.get<T>(), B.get<T>(), D.get<T>());

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          int ind1 = c(j, k)->addr.eval(i, N);
          int ind2 = (i / simd_width * dim * dim + j * dim + k) * simd_width +
                     i % simd_width;
          auto a = C.get<float32>()[ind1];
          auto b = D.get<T>()[ind2];
          if (std::abs(a - b) >= 1e-5_f) {
            TC_P(a);
            TC_P(b);
          }
          TC_ASSERT(std::abs(a - b) < 1e-5_f);
        }
      }
    }
  }

  return cpe;
}

template <int dim, typename T>
real TlangVec8AOSOA_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CPUCodeGen::Mode::vector, 8, 0);
}

template <int dim, typename T>
real TlangVec8Inter_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CPUCodeGen::Mode::vector, 8, 1);
}

template <int dim, typename T>
real TlangSca8_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CPUCodeGen::Mode::scalar, 8);
}

template <int dim, typename T>
real TlangVec16_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CPUCodeGen::Mode::vector, 16);
}

template <int dim, typename T>
real TlangSca16_matmatmul() {
  return Tlang_matmatmul<dim, T>(Tlang::CPUCodeGen::Mode::scalar, 16);
}
#define BENCHMARK(x)                                        \
  {                                                         \
    real t = x##_matmatmul<dim, T>();                       \
    fmt::print("  {:18s} = {:10.3f} cyc / elem \n", #x, t); \
  }

template <int dim, typename T>
void run_matmatmul() {
  fmt::print("Matrix<{}, {}>:\n", dim, sizeof(T) == 4 ? "float32" : "float64");
  BENCHMARK(TlangVec8AOSOA);
  BENCHMARK(TlangVec8Inter);
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

auto tlang_matmatmul = []() {
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

  run_matmatmul<2, float32>();
  run_matmatmul<4, float32>();
  // run_matmatmul<8, float32>();
};
TC_REGISTER_TASK(tlang_matmatmul);

void test_vec_add() {
  using namespace Tlang;
  constexpr int n = 16;

  CodeGen cg;
  auto &alloc = cg.alloc;
  Expr a;
  Expr b;
  alloc.buffer(0).stream(0).group().place(a);
  alloc.buffer(1).stream(0).group().place(b);
  auto c = a + b;
  Expr ret;
  c = ret.store(c);
  alloc.buffer(2).stream(0).group().place(c);
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
  /*
  fmt::print("   {:10.3f} cyc / elem      [adjusted run time = {:10.3f} ms]
  \n",
             cpu_frequency * 1e9 * t / elements, t * 1000.0_f);
             */
  fmt::print("   {:10.3f} cyc / elem  \n", cpu_frequency * 1e9 * t / elements);
}

void print_cpe(float64 cpe) {
  fmt::print("   {:10.3f} cyc / elem  \n", cpe);
}

template <int dim>
void test_mat_vec_mul_eigen(int in_cache) {
  fmt::print("dim={} eigen in_cache={}                      ", dim, in_cache);
  using namespace Tlang;

  int enlarge = in_cache ? 1 : 4096;
  int64 n = taichi::N * enlarge;

  EigenVector<Eigen::Matrix<float32, dim, dim>> m(n);
  EigenVector<Eigen::Matrix<float32, dim, 1>> v(n);
  EigenVector<Eigen::Matrix<float32, dim, 1>> mv(n);

  print_cpe(measure_cpe(
      [&]() {
        for (int i = 0; i < n; i++) {
          mv[i] = m[i] * v[i];
        }
      },
      n));
}

template <int dim>
void test_mat_vec_mul(int layout, int in_cache, int unroll, int prefetch) {
  std::string layout_name = "";
  if (layout == 0) {
    layout_name = "  soa";
  } else if (layout == 1) {
    layout_name = "aosoa";
  } else {
    layout_name = "inter";
  }
  fmt::print("dim={} {} in_cache={} unroll={} prefetch={:2d} ", dim,
             layout_name, in_cache, unroll, prefetch);
  using namespace Tlang;
  constexpr int simd_width = 8;
  CodeGen cg;
  cg.unroll = unroll;
  cg.prefetch = prefetch;
  auto &alloc = cg.alloc;
  auto &buffer = alloc.buffer(0);
  Matrix m(dim, dim), v(dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Address addr;
      addr.buffer_id = 0;
      if (layout == 0) {
        buffer.stream().group().place(m(i, j));
      } else if (layout == 1) {
        buffer.stream(0)
            .group(0)
            .group(i * dim + j)
            .repeat(simd_width)
            .place(m(i, j));
      } else {
        buffer.stream(0).group(j).repeat(simd_width / dim).place(m(i, j));
      }
    }
    Address addr;
    addr.buffer_id = 1;
    if (layout == 0) {
      alloc.buffer(1).stream().group().place(v(i));
    } else if (layout == 1) {
      alloc.buffer(1).stream(0).group(i).repeat(simd_width).place(v(i));
    } else {
      alloc.buffer(1).stream(0).group(0).repeat(simd_width / dim).place(v(i));
    }
  }
  Expr ret;
  auto mv = m * v;
  for (int i = 0; i < dim; i++) {
    Address addr;
    addr.buffer_id = 2;
    mv(i) = ret.store(mv(i));
    if (layout == 0) {
      alloc.buffer(2).stream().group().place(mv(i));
    } else if (layout == 1) {
      alloc.buffer(2).stream(0).group(i).repeat(simd_width).place(mv(i));
    } else {
      alloc.buffer(2).stream(0).group(0).repeat(simd_width / dim).place(mv(i));
    }
  }

  // alloc.print();

  int64 enlarge = in_cache ? 1 : 4096;
  int64 n = taichi::N * enlarge;
  TC_ASSERT(8 % dim == 0);
  int bs = 1;
  if (layout == 2) {  // interleaved
    bs = dim;
  }
  auto func = cg.get(ret, bs);

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

  print_cpe(measure_cpe(
      [&]() {
        func(M_allocator.get<float32>(), V_allocator.get<float32>(),
             MV_allocator.get<float32>(), n);
      },
      n));

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
  for (auto in_cache : {0, 1}) {
    test_mat_vec_mul_eigen<dim>(in_cache);
    for (auto layout : {0, 1, 2}) {
      for (auto unroll : {1, 4})
        for (auto prefetch : {0})
          test_mat_vec_mul<dim>(layout, in_cache, unroll, prefetch);
    }
    fmt::print("\n");
  }
}

auto tlang_matvecmul = []() {
  test_vec_add();
  test_mat_vec_mul_all<1>();
  test_mat_vec_mul_all<2>();
  test_mat_vec_mul_all<4>();
};
TC_REGISTER_TASK(tlang_matvecmul);

auto tlang_test = []() {
  default_measurement_time = 0;
  tlang_matvecmul();
  tlang_matmatmul();
};
TC_REGISTER_TASK(tlang_test);

auto tlang_benchmark = []() {
  default_measurement_time = 2;
  tlang_matvecmul();
  tlang_matmatmul();
};
TC_REGISTER_TASK(tlang_benchmark);

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

    addr.buffer_id = 0;
    vec_a(i) = load(addr);

    addr.buffer_id = 1;
    vec_b(i) = load(addr);
  }
  Matrix vec_c = vec_a + vec_b;
  for (int i = 0; i < M; i++) {
    Address addr;
    addr.coeff_i = 1;
    addr.coeff_const = i;

    addr.buffer_id = 2;
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
  constexpr int NUMPERPAGE = 512;  // # of elements to fit a page
  auto N = (int)size / 8;
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

auto allocator_test = []() {
  using namespace Tlang;
  {
    MemoryAllocator alloc;
    auto &buffer = alloc.buffer(0);
    auto &bundle = buffer.stream().group().repeat(4);
    Expr A, B, C;
    bundle.place(A, B);
    buffer.stream().group().place(C);
    alloc.materialize();
    TC_P(A->addr);
    TC_P(B->addr);
    TC_P(C->addr);
  }
  {
    MemoryAllocator alloc;
    auto &buffer = alloc.buffer(0);
    auto &g = buffer.stream();
    Expr A, B, C, D;
    g.group().repeat(4).place(A, C);
    g.group().repeat(4).place(B, D);
    alloc.materialize();
    TC_P(A->addr);
    TC_P(B->addr);
    TC_P(C->addr);
    TC_P(D->addr);
  }
};

TC_REGISTER_TASK(allocator_test);

/*
auto test_saxpy = []() {
  // fmt::print("dim={} {} in_cache={} unroll={} prefetch={:2d} ",);
  using namespace Tlang;
  CodeGen cg;
  auto alloc = cg.alloc;
  auto &buffer = alloc.buffer(0);

  int64 enlarge = 4096;
  int64 n = taichi::N * enlarge;
  int64 rounds = taichi::rounds / enlarge;
  cg.unroll = 4;
  cg.prefetch = 0;

  Expr ret;
  alloc.buffer(1).stream(0).place()

  AlignedAllocator A_allocator(n * sizeof(float32)),
      B_allocator(n * sizeof(float32));

  auto func = cg.get(ret);
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
    auto computed = MV_allocator.get<float32>()[mv(j)->addr.eval(i, n)];
  }
};
*/

TC_NAMESPACE_END

/*
TODO:
 CUDA backend
 Address Node
 imm
 vec3
 why eigen large variance
 why eigen so slow
 check unplaced variable
 auto batch sorting
 */
