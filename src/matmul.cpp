#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math.h>
#include <taichi/system/timer.h>
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
};

class AlignedAllocator {
  std::vector<uint8> _data;

  void *data;

  AlignedAllocator(int size) {
    _data.resize(size + 4096);
  }
};

// array of dim * dim * 8 * float32
template <int dim>
void AOSOA_matmul(float32 *A, float32 *B, float32 *C) {
  constexpr int simd_width = 8;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256 a[dim * dim], b[dim * dim];
      const int p = dim * dim * simd_width * t;
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_loadu_ps(&A[p + simd_width * i]);
        b[i] = _mm256_loadu_ps(&B[p + simd_width * i]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256 c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_storeu_ps(&C[p + simd_width * (i * dim + j)], c);
        }
      }
    }
  }
}

template <int dim, typename T>
real AOSOA_matmatmul() {
  std::vector<T> A, B, C;
  A.resize(N * dim * dim);
  B.resize(N * dim * dim);
  C.resize(N * dim * dim);

  auto t = Time::get_time();
  AOSOA_matmul<dim>(A.data(), B.data(), C.data());
  return Time::get_time() - t;
};

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
  BENCHMARK(AOSOA);
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
