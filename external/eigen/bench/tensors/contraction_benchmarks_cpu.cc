#define EIGEN_USE_THREADS

#include <string>

#include "tensor_benchmarks.h"

#define CREATE_THREAD_POOL(threads)             \
Eigen::ThreadPool pool(threads);                \
Eigen::ThreadPoolDevice device(&pool, threads);


// Contractions for number of threads ranging from 1 to 32
// Dimensions are Rows, Cols, Depth
#define BM_ContractionCPU(D1, D2, D3)                                         \
  static void BM_##Contraction##_##D1##x##D2##x##D3(int iters, int Threads) { \
    StopBenchmarkTiming();                                                    \
    CREATE_THREAD_POOL(Threads);                                              \
    BenchmarkSuite<Eigen::ThreadPoolDevice, float> suite(device, D1, D2, D3); \
    suite.contraction(iters);                                                 \
  }                                                                           \
  BENCHMARK_RANGE(BM_##Contraction##_##D1##x##D2##x##D3, 1, 32);


// Vector Matrix and Matrix Vector products
BM_ContractionCPU(1, 2000, 500);
BM_ContractionCPU(2000, 1, 500);

// Various skinny matrices
BM_ContractionCPU(250, 3, 512);
BM_ContractionCPU(1500, 3, 512);

BM_ContractionCPU(512, 800, 4);
BM_ContractionCPU(512, 80, 800);
BM_ContractionCPU(512, 80, 13522);
BM_ContractionCPU(1, 80, 13522);

BM_ContractionCPU(3200, 512, 4);
BM_ContractionCPU(3200, 512, 80);
BM_ContractionCPU(3200, 80, 512);
