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

constexpr int m = 128;

__device__ int indirect(int *c, int i) {
  return c[i % m] + i / m * m;
}

__global__ void fd(int *a, int *b, int *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  if (i > 0)
    sum += a[indirect(c, i) - 1];
  sum += a[indirect(c, i)];
  if (i < n - 1)
    sum += a[indirect(c, i) + 1];
  // b[i] = sqrt(sum) * 0.3f;
  atomicAdd(b + i % 128, sqrt(sum));
}

int main() {
  int n = 512 * 1024 * 1024;
  int *a, *b, *c;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&c, m * sizeof(float));
  for (int i = 0; i < n; i++) {
    a[i] = rand() * 1e-5f;
  }
  for (int i = 0; i < n; i++) {
    b[i] = rand() * 1e-5f;
  }
  for (int i = 0; i < m; i++) {
    c[i] = i;
  }
  for (auto bs : {32, 64, 128, 256, 512, 1024}) {
    std::cout << "bs = " << bs << std::endl;
    for (int i = 0; i < 16; i++) {
      auto t = get_time();
      fd<<<n / bs, bs>>>(a, b, c, n);
      cudaDeviceSynchronize();
      t = get_time() - t;
      printf("%.2f ms   bw %.3f GB/s\n", t * 1000,
             n * 2.0f * 4 / t / (1024 * 1024 * 1024.0f));
    }
    std::cout << std::endl;
  }
}
