#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#include "get_time.h"

constexpr int N = 4096;
constexpr int bs = 16;

__global__ void fill(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  a[i * N + j] = sinf(i * i - j * i + j);
}

__global__ void laplace(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (bs <= i && i < N - bs && bs <= j && j < N - bs) {
    auto ret = -4 * a[i * N + j] + a[i * N + j - 1] + a[i * N + j + 1] +
               a[i * N + j - N] + a[i * N + j + N];
    b[i * N + j] = ret;
  }
}

__global__ void laplace_shared(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int tid = blockDim.y * threadIdx.x + threadIdx.y;

  __shared__ float pad[bs + 2][bs + 2];
  auto pad_size = (bs + 2) * (bs + 2);

  if (bs <= i && i < N - bs && bs <= j && j < N - bs) {
    while (tid < pad_size) {
      int si = tid / (bs + 2);
      int sj = tid % (bs + 2);
      int gi = si - 1 + blockIdx.x * blockDim.x;
      int gj = sj - 1 + blockIdx.y * blockDim.y;
      pad[si][sj] = a[gi * N + gj];
      tid += blockDim.x * blockDim.y;
    }
  }

  __syncthreads();

  if (bs <= i && i < N - bs && bs <= j && j < N - bs) {
    auto ret = -4 * pad[threadIdx.x + 1][threadIdx.y + 1] +
               pad[threadIdx.x + 2][threadIdx.y + 1] +
               pad[threadIdx.x][threadIdx.y + 1] +
               pad[threadIdx.x + 1][threadIdx.y + 2] +
               pad[threadIdx.x + 1][threadIdx.y];
    b[i * N + j] = ret;
  }
}

__global__ void compare(float *a, float *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (fabs(a[i * N + j] - b[i * N + j]) > 1e-5) {
    printf("error! %d %d %f %f\n", i, j, a[i * N + j], b[i * N + j]);
  }
}

int main() {
  float *a, *b, *c;
  cudaMalloc(&a, N * N * sizeof(float));
  cudaMalloc(&b, N * N * sizeof(float));
  cudaMalloc(&c, N * N * sizeof(float));

  cudaMemset(a, 0, N * N * sizeof(float));
  cudaMemset(b, 0, N * N * sizeof(float));
  cudaMemset(c, 0, N * N * sizeof(float));

  int repeat = 25;

  double t;
  dim3 grid_dim(N / bs, N / bs, 1);
  dim3 block_dim(bs, bs, 1);

  fill<<<grid_dim, block_dim>>>(a);

  cudaDeviceSynchronize();
  t = get_time();
  for (int i = 0; i < repeat; i++) {
    laplace<<<grid_dim, block_dim>>>(a, b);
  }
  cudaDeviceSynchronize();
  t = (get_time() - t) / repeat;
  printf("no shared block_dim %d, %.2f ms\n", bs, t * 1000);

  cudaDeviceSynchronize();
  t = get_time();
  for (int i = 0; i < repeat; i++) {
    laplace_shared<<<grid_dim, block_dim>>>(a, c);
  }
  cudaDeviceSynchronize();
  t = (get_time() - t) / repeat;
  printf("with shared block_dim %d, %.2f ms\n", bs, t * 1000);

  compare<<<grid_dim, block_dim>>>(b, c);
  cudaDeviceSynchronize();
}

// make laplace && nvprof --print-gpu-trace ./laplace
