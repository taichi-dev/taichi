#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include "get_time.h"

__global__ void cpy(float *a, float *b, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = b[i];
}

int main() {
  int n = 1024 * 1024 * 1024;
  float *a, *b;
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

  for (auto bs : {32, 64, 128, 256}) {
    for (int i = 0; i < 10; i++) {
      cpy<<<n / bs, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    t = get_time();
    for (int i = 0; i < repeat; i++) {
      cpy<<<n / bs, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    t = (get_time() - t) / repeat;
    printf("memcpy 8GB data, block_dim %d, %.2f ms   bw %.3f GB/s\n", bs,
           t * 1000, n * 8.0 / t / (1024 * 1024 * 1024.0f));
  }
}
