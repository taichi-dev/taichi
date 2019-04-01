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

inline __device__ int indirect(int *c, int i) {
  // return c[c[i & 127] & 127] + i;
  return int(exp(((((float(i))))) * 1e-18)) + i;
  // printf("%d\n", c[i % m]  - i % m + i - i);
  // return i;
}

__constant__ int const_c[m];

__global__ void fd(float *a, float *b, int *c, int n) {
  // __shared__ int b_s[m];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /*
  if (threadIdx.x < m) {
    b_s[threadIdx.x] = c[threadIdx.x];
  }
  __syncthreads();
   */
  /*
  float sum = 0;
  if (i > 0)
    sum += a[indirect(c, i) - 1];
  */
  // sum += a[indirect(c, i)];
  // sum += a[i + b_s[i & 127]];
  /*
  if (i < n - 1)
    sum += a[indirect(c, i) + 1];
  */
  // b[i] = (i * 1e-18);
  // b[i] = i;
  b[i] = c[c[c[i & 64]]];
  /*
  atomicAdd(&b_s[0], sqrt(sum));
  if (threadIdx.x < m) {
    atomicAdd(b + threadIdx.x, b_s[threadIdx.x]);
    // b[threadIdx.x] += b_s[threadIdx.x];
  }
  */
}

int main() {
  int n = 1024 * 1024 * 1024;
  float *a, *b;
  int *c;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&c, m * sizeof(float));
  for (int i = 0; i < n; i++) {
    a[i] = i * 1e-5f;
  }
  for (int i = 0; i < n; i++) {
    b[i] = i * 1e-5f;
  }
  for (int i = 0; i < m; i++) {
    c[i] = i;
  }
  cudaMemcpyToSymbol(const_c, c, m * sizeof(float), 0, cudaMemcpyHostToDevice);
  for (auto bs : {128, 256, 512, 1024}) {
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
