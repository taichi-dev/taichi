#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/system/timer.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "tlang.h"

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
real AOS_matmatmul(std::size_t N) {
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

// array of N * dim * dim * 8 * float32
template <int dim>
real AOSOA_matmul(std::size_t N, float32 *A, float32 *B, float32 *C) {
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

template <int dim, typename T>
real AOSOA_AVX2_matmatmul(std::size_t N) {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  return AOSOA_matmul<dim>(N, A.get<T>(), B.get<T>(), C.get<T>());
};

template <int dim, typename T>
real SOA_AVX2_matmatmul(std::size_t N) {
  AlignedAllocator A(sizeof(T) * N * dim * dim);
  AlignedAllocator B(sizeof(T) * N * dim * dim);
  AlignedAllocator C(sizeof(T) * N * dim * dim);

  constexpr int simd_width = 8;
  auto task = [&]() {
    for (int t = 0; t < N / simd_width; t++) {
      __m256 a[dim * dim], b[dim * dim];
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_ps(&A.get<float32>()[i * N + t * simd_width]);
        b[i] = _mm256_load_ps(&B.get<float32>()[i * N + t * simd_width]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256 c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_ps(&C.get<float32>()[(i * dim + j) * N + t * simd_width],
                          c);
        }
      }
    }
  };
  return measure_cpe(task, N);
};

template <int dim, typename T>
real Tlang_matmatmul(std::size_t N, Arch arch, int layout, int in_cache) {
  Matrix a(dim, dim), b(dim, dim), c(dim, dim);

  int simd_width = default_simd_width(arch);

  int n = N;

  Program prog(arch);
  int scale = 1;
  // prog.config.group_size = layout == 1 ? dim : 1;
  // prog.config.num_groups = 8 / prog.config.group_size * scale;

  // TODO: eliminate this
  layout = 2;
  auto in = ind();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a(i, j) = var<float32>();
      b(i, j) = var<float32>();
      c(i, j) = var<float32>();
    }
  }

  prog.layout([&]() {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (layout == 0) {
          // AOSOA
          /*
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
           */
        } else if (layout == 1) {  // Inter
          /*
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
              */
        } else {  // SOA
          root.fixed(in, n).place(a(i, j));
          root.fixed(in, n).place(b(i, j));
          root.fixed(in, n).place(c(i, j));
        }
      }
    }
  });

  TC_P(n);

  auto mul = kernel(c(0, 0), [&]() { c[in] = a[in] * b[in]; });

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
        a(j, k).val<float32>(i) = A.get<float32>()[ind2];
        b(j, k).val<float32>(i) = B.get<float32>()[ind2];
      }
    }
  }

  auto cpe = measure_cpe([&]() { mul(); }, n);

  AOSOA_matmul<dim>(N, A.get<T>(), B.get<T>(), D.get<T>());

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        int ind2 = (i / 8 * dim * dim + j * dim + k) * 8 + i % 8;
        auto a = c(j, k).val<float32>(i);
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
real TlangGPUAOSOA_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::gpu, 0, 0);
}

template <int dim, typename T>
real TlangGPUInter_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::gpu, 1, 0);
}

template <int dim, typename T>
real TlangGPUSOA_matmatmul(std::size_t N) {
  return Tlang_matmatmul<dim, T>(N, Arch::gpu, 2, 0);
}

#define BENCHMARK(x)                                        \
  {                                                         \
    real t = x##_matmatmul<dim, T>(512);                     \
    fmt::print("  {:18s} = {:10.3f} cyc / elem \n", #x, t); \
  }

template <int dim, typename T>
void run_matmatmul() {
  fmt::print("Matrix<{}, {}>:\n", dim, sizeof(T) == 4 ? "float32" : "float64");

  // BENCHMARK(TlangGPUAOSOA);
  // BENCHMARK(TlangGPUInter);
  // BENCHMARK(TlangGPUSOA);

  // BENCHMARK(TlangCPUAOSOA);
  // BENCHMARK(TlangCPUInter);
  BENCHMARK(TlangCPUSOA);

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

  run_matmatmul<1, float32>();
  run_matmatmul<2, float32>();
  run_matmatmul<4, float32>();
  run_matmatmul<8, float32>();
};
TC_REGISTER_TASK(tlang_matmatmul);

void print_cpe(float64 cpe) {
  fmt::print(" {:10.3f} cyc / elem\n", cpe);
}

template <int dim>
void test_mat_vec_mul_eigen(int in_cache) {
  fmt::print("dim={} eigen in_cache={}          ", dim, in_cache);

  int enlarge = in_cache ? 1 : 4096;
  int64 n = 256 * enlarge;

  EigenVector<Eigen::Matrix<float32, dim, dim>> m(
      n, Eigen::Matrix<float32, dim, dim>::Ones());
  EigenVector<Eigen::Matrix<float32, dim, 1>> v(
      n, Eigen::Matrix<float32, dim, 1>::Ones());
  EigenVector<Eigen::Matrix<float32, dim, 1>> mv(
      n, Eigen::Matrix<float32, dim, 1>::Ones());

  print_cpe(measure_cpe(
      [&]() {
        for (int i = 0; i < n; i++) {
          mv[i] = m[i] * v[i];
        }
      },
      n));
}

template <int dim>
void test_mat_vec_mul(Arch arch, int layout, int in_cache) {
#if (0)
  std::string layout_name = "";
  if (layout == 0) {
    layout_name = "  soa";
  } else if (layout == 1) {
    layout_name = "aosoa";
  } else {
    layout_name = "inter";
  }
  fmt::print("dim={} {} in_cache={} arch={} ", dim, layout_name, in_cache,
             arch == Arch::x86_64 ? "CPU" : "GPU");
  int simd_width = default_simd_width(arch);

  int64 enlarge = in_cache ? 1 : 4096;
  int64 n = 256 * enlarge;
  Program prog(arch, n);
  // cg.unroll = unroll;
  // cg.prefetch = prefetch;
  auto &alloc = prog;
  auto &buffer = alloc.buffer(0);
  Matrix m(dim, dim), v(dim), mv(dim, 1);
  prog.layout([&]() {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
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
      if (layout == 0) {
        alloc.buffer(1).stream().group().place(v(i));
      } else if (layout == 1) {
        alloc.buffer(1).stream(0).group(i).repeat(simd_width).place(v(i));
      } else {
        alloc.buffer(1).stream(0).group(0).repeat(simd_width / dim).place(v(i));
      }
    }
    for (int i = 0; i < dim; i++) {
      if (layout == 0) {
        alloc.buffer(2).stream().group().place(mv(i));
      } else if (layout == 1) {
        alloc.buffer(2).stream(0).group(i).repeat(simd_width).place(mv(i));
      } else {
        alloc.buffer(2)
            .stream(0)
            .group(0)
            .repeat(simd_width / dim)
            .place(mv(i));
      }
    }
  });

  TC_ASSERT(8 % dim == 0);
  int bs = 1;
  if (layout == 2) {  // interleaved
    bs = dim;
  }
  prog.config.simd_width = 8;
  prog.config.group_size = bs;

  auto mul = prog.def([&]() {
    Index ind = Expr::index(0);
    mv[ind] = m[ind] * v[ind];
  });

  std::vector<Eigen::Matrix<float32, dim, 1>> ground_truth(n);
  for (int i = 0; i < n; i++) {
    Eigen::Matrix<float32, dim, dim> m_gt;
    Eigen::Matrix<float32, dim, 1> v_gt;
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        m_gt(j, k) = rand<float32>();
        prog.data(m(j, k), i) = m_gt(j, k);
      }
      v_gt(j) = rand<float32>();
      prog.data(v(j), i) = v_gt(j);
    }
    Eigen::Matrix<float32, dim, 1> mv_gt = m_gt * v_gt;
    ground_truth[i] = mv_gt;
  }

  print_cpe(measure_cpe([&]() { mul(); }, n));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      auto computed = prog.data(mv(j), i);
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
#endif
}

template <int dim>
void test_mat_vec_mul_all() {
  for (auto in_cache : {0, 1}) {
    test_mat_vec_mul_eigen<dim>(in_cache);
    for (auto layout : {0, 1, 2}) {
      std::vector<Arch> archs;
      if (in_cache) {
        archs = {Arch::x86_64};
      } else {
        // archs = {Arch::x86_64, Arch::gpu};
        archs = {Arch::x86_64};
      }
      for (auto arch : archs)
        test_mat_vec_mul<dim>(arch, layout, in_cache);
    }
    fmt::print("\n");
  }
}

auto tlang_matvecmul = []() {
  initialize_benchmark();
  test_mat_vec_mul_all<1>();
  test_mat_vec_mul_all<2>();
  test_mat_vec_mul_all<4>();
  test_mat_vec_mul_all<8>();
};
TC_REGISTER_TASK(tlang_matvecmul);

auto tlang_test = []() {
  default_measurement_time = 0;
  tlang_matmatmul();
  tlang_matvecmul();
};
TC_REGISTER_TASK(tlang_test);

auto tlang_benchmark = []() {
  default_measurement_time = 2;
  tlang_matmatmul();
  tlang_matvecmul();
};
TC_REGISTER_TASK(tlang_benchmark);

TC_NAMESPACE_END

/*
TODO:
 alert if address assigned multiple times
 imm
 vec3
 check unplaced variable
 assert n % 256 = 0
*/
