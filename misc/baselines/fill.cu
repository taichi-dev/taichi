#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int m = 256;
constexpr int n = m * m * m;
constexpr int block_size = 4;

using grid_type = float[m / block_size][m / block_size][m / block_size][4]
                       [block_size][block_size][block_size];

__global__ void fill(grid_type *grid_) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float *data = (float *)grid_;
  grid_type &grid = *grid_;
  for (int k = 0; k < 4; k++) {
    data[i + k * n] = 0;
  }
}

int main() {
  float *a;
  cudaMallocManaged(&a, n * sizeof(float) * 4);
  auto bs = block_size * block_size * block_size;
  std::cout << "bs = " << bs << std::endl;
  for (int i = 0; i < 20; i++) {
    cudaDeviceSynchronize();
    auto t = get_time();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaDeviceSynchronize();
    fill<<<n / bs, bs>>>((grid_type *)a);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "device  " << milliseconds << std::endl;
  }
  std::cout << std::endl;
}
