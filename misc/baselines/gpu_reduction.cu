#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include "get_time.h"

__inline__ __device__ int warpReduceSum(int val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(val, offset, 0xFFFFFFFF);
  return val;
}

__global__ void cpy(int *a, int *b, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  while (i < n) {
    int val = b[i];
    sum += val;
    i += blockDim.x * gridDim.x;
  }
  atomicAdd(a, sum);
}

int main() {
  int n = 1024 * 1024 * 1024 / 4;
  int *a, *b;
  cudaMalloc(&a, n * sizeof(float));
  cudaMalloc(&b, n * sizeof(float));

  int repeat = 25;

  double t;
  t = get_time();
  for (int i = 0; i < repeat; i++) {
    cudaMemcpyAsync(a, b, n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
  }
  cudaDeviceSynchronize();
  t = (get_time() - t) / repeat;
  printf("cuMemcpyAsync 8GB data bw %.3f GB/s\n",
         n * 8.0 / t / (1024 * 1024 * 1024.0f));

  for (auto bs : {32, 64, 128, 256, 512, 1024}) {
    for (int i = 0; i < 10; i++) {
      cpy<<<896, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    t = get_time();
    for (int i = 0; i < repeat; i++) {
      cpy<<<896, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    t = (get_time() - t) / repeat;
    printf("reducing 4 GB data, block_dim %d, %.2f ms   bw %.3f GB/s\n", bs,
           t * 1000, n * 4.0 / t / (1024 * 1024 * 1024.0f));
  }
}
