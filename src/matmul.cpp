#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math.h>
#include <taichi/system/timer.h>
#include <Eigen/Dense>

TC_NAMESPACE_BEGIN

constexpr int rounds = 8192;
constexpr int N = 1024;

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

#define BENCHMARK(x)                                                      \
  {                                                                       \
    real t = x##_matmatmul<dim, T>();                                     \
    fmt::print("Matrix<{}, {}>, {} = {:8.3f} ms\n", dim,                     \
               sizeof(T) == 4 ? "float32" : "float64", #x, t * 1000.0_f); \
  }

template <int dim, typename T>
void run() {
  BENCHMARK(eigen);
}

auto benchmark_matmul = []() {
  run<2, float32>();
  run<3, float32>();
  run<4, float32>();
  run<2, float64>();
  run<3, float64>();
  run<4, float64>();
};
TC_REGISTER_TASK(benchmark_matmul);

TC_NAMESPACE_END
