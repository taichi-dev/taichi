#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include "get_time.h"

constexpr int N = 4096;
constexpr int bs = 4;

__global__ void laplace(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (bs <= i && i < N - bs && bs <= j && j < N - bs) {
    auto ret = -4 * a[i * N + j] + a[i * N + j - 1] + a[i * N + j + 1] +
               a[i * N + j - N] + a[i * N + j + N];
    b[i * N + j] = ret;
  }
}

/*
__global__ void laplace(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (bs <= i && i < N - bs && bs <= j && j < N - bs) {
    auto ret = -4 * a[i * N + j] + a[i * N + j - 1] + a[i * N + j + 1] +
               a[i * N + j - N] + a[i * N + j + N];
    b[i * N + j] = ret;
  }
}
*/

int main() {
  int *a, *b;
  cudaMalloc(&a, N * N * sizeof(float));
  cudaMalloc(&b, N * N * sizeof(float));

  int repeat = 25;

  double t;
  cudaDeviceSynchronize();
  t = get_time();
  dim3 grid_dim(N / bs, N / bs, 1);
  dim3 block_dim(bs, bs, 1);
  for (int i = 0; i < repeat; i++) {
    laplace<<<grid_dim, block_dim>>>(a, b);
  }
  cudaDeviceSynchronize();
  t = (get_time() - t) / repeat;
  printf("block_dim %d, %.2f ms\n", bs, t * 1000);
}
