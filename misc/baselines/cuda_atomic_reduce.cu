#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include "get_time.h"

__global__ void reduce(int *a, int *b, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(b, a[i]);
}

int main() {
  // int n = 512 * 512;
  int n = 512 * 512;
  int *a, *b;
  cudaMalloc(&a, n * sizeof(int));
  cudaMalloc(&b, sizeof(int));
  for (auto bs : {16, 32, 64, 128, 256}) {
    for (int i = 0; i < 10; i++) {
      reduce<<<n / bs, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    int repeat = 1000;
    auto t = get_time();
    for (int i = 0; i < repeat; i++) {
      reduce<<<n / bs, bs>>>(a, b, n);
    }
    cudaDeviceSynchronize();
    t = (get_time() - t) / repeat;
    printf("atomic reduce, block_size %d, %.4f ms\n", bs, t * 1000);
  }
}
