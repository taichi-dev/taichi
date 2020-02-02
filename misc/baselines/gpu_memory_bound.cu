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

__global__ void cpy(float *a, float *b, int *c, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = b[i];
}

int main() {
  int n = 1024 * 1024 * 1024 / 4;
  float *a, *b;
  int *c;
  cudaMalloc(&a, n * sizeof(float));
  cudaMalloc(&b, n * sizeof(float));
  cudaMalloc(&c, n * sizeof(float));
  /*
  for (int i = 0; i < n; i++) {
    a[i] = i * 1e-5f;
  }
  for (int i = 0; i < n; i++) {
    b[i] = i * 1e-5f;
  }
  for (int i = 0; i < n; i++) {
    c[i] = 0;
  }
   */
  for (auto bs : {16, 32, 64, 128, 256}) {
    for (int i = 0; i < 10; i++) {
      cpy << < n / bs, bs >> > (a, b, c, n);
    }
    cudaDeviceSynchronize();
    int repeat = 100;
    auto t = get_time();
    for (int i = 0; i < repeat; i++) {
      cpy<<<n / bs, bs>>>(a, b, c, n);
    }
    cudaDeviceSynchronize();
    t = (get_time() - t) / repeat;
    printf("memcpy 1GB data, block_size %d, %.2f ms   bw %.3f GB/s\n", bs, t * 1000,
           n * 8.0 / t / (1024 * 1024 * 1024.0f));
  }
}
