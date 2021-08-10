#define EIGEN_USE_SYCL

#include <SYCL/sycl.hpp>
#include <iostream>

#include "tensor_benchmarks.h"

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;
// Simple functions
template <typename device_selector>
cl::sycl::queue sycl_queue() {
  return cl::sycl::queue(device_selector(), [=](cl::sycl::exception_list l) {
    for (const auto& e : l) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception e) {
        std::cout << e.what() << std::endl;
      }
    }
  });
}

#define BM_FuncGPU(FUNC)                                       \
  static void BM_##FUNC(int iters, int N) {                    \
    StopBenchmarkTiming();                                     \
    cl::sycl::queue q = sycl_queue<cl::sycl::gpu_selector>();  \
    Eigen::SyclDevice device(q);                               \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, N); \
    suite.FUNC(iters);                                         \
  }                                                            \
  BENCHMARK_RANGE(BM_##FUNC, 10, 5000);

BM_FuncGPU(broadcasting);
BM_FuncGPU(coeffWiseOp);
