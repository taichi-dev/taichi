#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/system/timer.h>
#include <Eigen/StdVector>
#include "tlang.h"
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

using namespace Tlang;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <int dim, typename T>
real AOS_eigen_matmatmul(std::size_t N) {
  using Matrix = Eigen::Matrix<T, dim, dim>;
  EigenVector<Matrix> A(N, Matrix::Zero()), B(N, Matrix::Ones()),
      C(N, Matrix::Ones());

  return measure_cpe(
      [&]() {
        for (int i = 0; i < N; i++) {
          C[i] = A[i] * B[i];
        }
      },
      N);
};
template <int dim, typename T>
real AOS_eigen_unroll2_matmatmul(std::size_t N) {
  using Matrix = Eigen::Matrix<T, dim, dim>;
  EigenVector<Matrix> A(N, Matrix::Zero()), B(N, Matrix::Ones()),
      C(N, Matrix::Ones());

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
real AOS_eigen_unroll4_matmatmul(std::size_t N) {
  using Matrix = Eigen::Matrix<T, dim, dim>;
  EigenVector<Matrix> A(N, Matrix::Zero()), B(N, Matrix::Ones()),
      C(N, Matrix::Ones());

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
real Tlang_matmatmul(std::size_t N, Arch arch, int layout, int in_cache) {
  Matrix a(dim, dim), b(dim, dim), c(dim, dim);

  int simd_width = default_simd_width(arch);

  int n = N;

  Program prog(arch);
  prog.config.group_size = 1; // layout == 1 ? dim : 1;
  int scale = 1;
  prog.config.num_groups = 8 / prog.config.group_size * scale;

  auto p = ind();

  layout([&]() {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        a(i, j) = var<float32>();
        b(i, j) = var<float32>();
        c(i, j) = var<float32>();
        root.fixed(p, n).place(a(i, j));
        root.fixed(p, n).place(b(i, j));
        root.fixed(p, n).place(c(i, j));
        /*
        if (layout == 0) {
          // AOSOA
          prog.buffer(0)
              .stream(0)
              .group(0)
              .group(i * dim + j)
              .repeat(simd_width * scale)
              .place(a(i, j));
          prog.buffer(1)
              .stream(0)
              .group(0)
              .group(i * dim + j)
              .repeat(simd_width * scale)
              .place(b(i, j));
          prog.buffer(2)
              .stream(0)
              .group(0)
              .group(i * dim + j)
              .repeat(simd_width * scale)
              .place(c(i, j));
        } else if (layout == 1) {  // Inter
          prog.buffer(0)
              .stream(0)
              .group(j)
              .repeat(simd_width / dim * scale)
              .place(a(i, j));
          prog.buffer(1)
              .stream(0)
              .group(j)
              .repeat(simd_width / dim * scale)
              .place(b(i, j));
          prog.buffer(2)
              .stream(0)
              .group(j)
              .repeat(simd_width / dim * scale)
              .place(c(i, j));
        } else {  // SOA
          prog.buffer(0).stream(i * dim + j).group().place(a(i, j));
          prog.buffer(1).stream(i * dim + j).group().place(b(i, j));
          prog.buffer(2).stream(i * dim + j).group().place(c(i, j));
        }
        */
      }
    }
  });

  auto mul = kernel(a(0, 0), [&]() {
    c[p] = a[p] * b[p];
  });

  /*
  AlignedAllocator A(sizeof(T) * n * dim * dim);
  AlignedAllocator B(sizeof(T) * n * dim * dim);
  AlignedAllocator D(sizeof(T) * n * dim * dim);

  for (int i = 0; i < N * dim * dim; i++) {
    A.get<T>()[i] = rand();
    B.get<T>()[i] = rand();
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        int ind2 = (i / 8 * dim * dim + j * dim + k) * 8 + i % 8;
        prog.data(a(j, k), i) = A.get<float32>()[ind2];
        prog.data(b(j, k), i) = B.get<float32>()[ind2];
      }
    }
  }
  */

  auto cpe = measure_cpe([&]() { mul(); }, n);

  /*
  AOSOA_matmul<dim>(N, A.get<T>(), B.get<T>(), D.get<T>());

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        int ind2 = (i / 8 * dim * dim + j * dim + k) * 8 + i % 8;
        auto a = prog.data(c(j, k), i);
        auto b = D.get<T>()[ind2];
        if (std::abs(a - b) >= 1e-5_f) {
          TC_P(a);
          TC_P(b);
          TC_P(i);
        }
        TC_ASSERT(std::abs(a - b) < 1e-5_f);
      }
    }
  }
  */

  return cpe;
}

template <int dim, typename T>
real TlangCPUAOSOA_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::x86_64, 0, 0);
}

template <int dim, typename T>
real TlangCPUInter_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::x86_64, 1, 0);
}

template <int dim, typename T>
real TlangCPUSOA_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::x86_64, 2, 0);
}

template <int dim, typename T>
void run_matmatmul() {
  fmt::print("Matrix<{}, {}>:\n", dim, sizeof(T) == 4 ? "float32" : "float64");

  // BENCHMARK(TlangGPUAOSOA);
  // BENCHMARK(TlangGPUInter);
  // BENCHMARK(TlangGPUSOA);

  BENCHMARK(TlangCPUAOSOA);
  BENCHMARK(TlangCPUInter);
  BENCHMARK(TlangCPUSOA);

  BENCHMARK(AOS_eigen);
  BENCHMARK(AOS_eigen_unroll2);
  BENCHMARK(AOS_eigen_unroll4);

  fmt::print("\n");
}

void initialize_benchmark() {
  // CoreState::set_trigger_gdb_when_crash(true);
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
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
  // TC_INFO("NVCC  Version {}.{}.{}", __CUDACC_VER_MAJOR__,
  // __CUDACC_VER_MINOR__,
  //        __CUDACC_VER_BUILD__);
}

auto tlang_matmatmul = []() {
  initialize_benchmark();

  run_matmatmul<2, float32>();
  run_matmatmul<4, float32>();
  run_matmatmul<8, float32>();
};
TC_REGISTER_TASK(tlang_matmatmul);

void print_cpe(float64 cpe) {
  fmt::print(" {:10.3f} cyc / elem\n", cpe);
}

auto benchmark_matmatmul = []() {
  default_measurement_time = 2;
  tlang_matmatmul();
};
TC_REGISTER_TASK(benchmark_matmatmul);

TC_NAMESPACE_END

/*
TODO:
 alert if address assigned multiple times
 imm
 vec3
 check unplaced variable
 assert n % 256 = 0
*/
