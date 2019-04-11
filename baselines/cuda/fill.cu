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
    fill<<<n / bs, bs>>>((grid_type *)a);
    cudaDeviceSynchronize();
    t = get_time() - t;
    printf("%.2f ms   bw %.3f GB/s\n", t * 1000, n * 1.0f * 16 / t / 1e9);
  }
  std::cout << std::endl;
}
